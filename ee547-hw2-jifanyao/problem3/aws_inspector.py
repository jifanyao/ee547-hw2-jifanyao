#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import json
import os
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError, NoRegionError


def retryable(fn):
    def wrapper(*args, **kwargs):
        delays = [0.5, 1.0]   
        try:
            return fn(*args, **kwargs)
        except (EndpointConnectionError,) as e:
            
            for d in delays:
                time.sleep(d)
                try:
                    return fn(*args, **kwargs)
                except EndpointConnectionError:
                    continue
            raise e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("Throttling", "ThrottlingException", "RequestLimitExceeded", "ServiceUnavailable", "RequestTimeout"):
                for d in delays:
                    time.sleep(d)
                    try:
                        return fn(*args, **kwargs)
                    except ClientError as e2:
                        code2 = e2.response.get("Error", {}).get("Code", "")
                        if code2 not in ("Throttling", "ThrottlingException", "RequestLimitExceeded", "ServiceUnavailable", "RequestTimeout"):
                            break
                raise e
            raise e
    return wrapper

def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def warn(msg: str):
    print(f"[WARNING] {msg}", file=sys.stderr)


def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)


def validate_region(region: Optional[str]) -> Optional[str]:

    if not region:
        return None
    session = boto3.session.Session()
    valid = set(session.get_available_regions("ec2"))
    if region not in valid:
        raise ValueError(f"Invalid region: {region}")
    return region


@retryable
def get_identity() -> Dict[str, str]:
    sts = boto3.client("sts")
    return sts.get_caller_identity()


@retryable
def _iam_list_users(iam):
    return iam.list_users()

@retryable
def _iam_get_user(iam, user_name: str):
    return iam.get_user(UserName=user_name)

@retryable
def _iam_list_attached_user_policies(iam, user_name: str):
    return iam.list_attached_user_policies(UserName=user_name)

def collect_iam_users() -> List[Dict[str, Any]]:
    data = []
    try:
        iam = boto3.client("iam")
        users = _iam_list_users(iam).get("Users", [])
        for u in users:
            username = u.get("UserName")
            try:
                g = _iam_get_user(iam, username)
                user = g.get("User", {})
                last_used = user.get("PasswordLastUsed")
                last_activity = last_used.replace(microsecond=0).isoformat() + "Z" if last_used else None
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("AccessDenied", "AccessDeniedException"):
                    warn(f"Access denied for iam:GetUser on '{username}' - skipping last_activity")
                    last_activity = None
                else:
                    raise

            attached = []
            try:
                pols = _iam_list_attached_user_policies(iam, username).get("AttachedPolicies", [])
                for p in pols:
                    attached.append({
                        "policy_name": p.get("PolicyName"),
                        "policy_arn": p.get("PolicyArn")
                    })
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("AccessDenied", "AccessDeniedException"):
                    warn(f"Access denied for iam:ListAttachedUserPolicies on '{username}' - policies omitted")
                else:
                    raise

            data.append({
                "username": username,
                "user_id": u.get("UserId"),
                "arn": u.get("Arn"),
                "create_date": u.get("CreateDate").replace(microsecond=0).isoformat() + "Z" if u.get("CreateDate") else None,
                "last_activity": last_activity,
                "attached_policies": attached
            })
    except NoCredentialsError:
        error("Authentication failed (no credentials).")
        sys.exit(1)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "AccessDeniedException"):
            warn("Access denied for IAM operations - skipping user enumeration")
            return []
        else:
            warn(f"IAM collection error: {e}")
            return []
    except EndpointConnectionError:
        warn("Network error when calling IAM - skipping users")
        return []
    return data


@retryable
def _ec2_describe_instances(ec2):
    return ec2.describe_instances()

@retryable
def _ec2_describe_images(ec2, image_ids: List[str]):
    return ec2.describe_images(ImageIds=image_ids)

@retryable
def _ec2_describe_security_groups(ec2):
    return ec2.describe_security_groups()

def _sg_rules_to_brief(rule: Dict[str, Any], direction: str) -> Dict[str, str]:
    ip_protocol = rule.get("IpProtocol", "-1")
    protocol = "all" if ip_protocol in ("-1", None) else ip_protocol

    from_p = rule.get("FromPort")
    to_p = rule.get("ToPort")
    if from_p is None and to_p is None:
        port_range = "all"
    else:
        if from_p == to_p:
            port_range = f"{from_p}-{to_p}"
        else:
            port_range = f"{from_p}-{to_p}"

    cidrs = []
    for c in rule.get("IpRanges", []):
        cidrs.append(c.get("CidrIp"))
    for c in rule.get("Ipv6Ranges", []):
        cidrs.append(c.get("CidrIpv6"))
    for p in rule.get("PrefixListIds", []):
        cidrs.append(p.get("PrefixListId"))
    for u in rule.get("UserIdGroupPairs", []):
        cidrs.append(u.get("GroupId"))

    if direction == "in":
        src_or_dst = "source"
    else:
        src_or_dst = "destination"
    endpoint = cidrs[0] if cidrs else "0.0.0.0/0"

    return {
        "protocol": protocol,
        "port_range": port_range,
        src_or_dst: endpoint
    }

def collect_ec2_and_sg(region: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    instances_out = []
    sgs_out = []
    try:
        ec2 = boto3.client("ec2", region_name=region)

        resp = _ec2_describe_instances(ec2)
        reservations = resp.get("Reservations", [])
        image_ids = set()
        for res in reservations:
            for ins in res.get("Instances", []):
                image_id = ins.get("ImageId")
                if image_id:
                    image_ids.add(image_id)

        ami_name_map = {}
        if image_ids:
            try:
                imgs = _ec2_describe_images(ec2, list(image_ids)).get("Images", [])
                for im in imgs:
                    ami_name_map[im.get("ImageId")] = im.get("Name")
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("AccessDenied", "AccessDeniedException"):
                    warn("Access denied for ec2:DescribeImages - AMI names omitted")
                else:
                    raise

        for res in reservations:
            for ins in res.get("Instances", []):
                state = ins.get("State", {}).get("Name")
                sg_ids = [g.get("GroupId") for g in ins.get("SecurityGroups", []) if g.get("GroupId")]
                tags = {t.get("Key"): t.get("Value") for t in ins.get("Tags", []) if t.get("Key")}
                instances_out.append({
                    "instance_id": ins.get("InstanceId"),
                    "instance_type": ins.get("InstanceType"),
                    "state": state,
                    "public_ip": ins.get("PublicIpAddress"),
                    "private_ip": ins.get("PrivateIpAddress"),
                    "availability_zone": ins.get("Placement", {}).get("AvailabilityZone"),
                    "launch_time": ins.get("LaunchTime").replace(microsecond=0).isoformat() + "Z" if ins.get("LaunchTime") else None,
                    "ami_id": ins.get("ImageId"),
                    "ami_name": ami_name_map.get(ins.get("ImageId")),
                    "security_groups": sg_ids,
                    "tags": tags
                })

        sgresp = _ec2_describe_security_groups(ec2).get("SecurityGroups", [])
        for sg in sgresp:
            inbound = [_sg_rules_to_brief(r, "in") for r in sg.get("IpPermissions", [])] or []
            outbound = [_sg_rules_to_brief(r, "out") for r in sg.get("IpPermissionsEgress", [])] or []
            sgs_out.append({
                "group_id": sg.get("GroupId"),
                "group_name": sg.get("GroupName"),
                "description": sg.get("Description"),
                "vpc_id": sg.get("VpcId"),
                "inbound_rules": inbound,
                "outbound_rules": outbound
            })

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "AccessDeniedException"):
            warn("Access denied for EC2 operations - skipping instances/security groups")
            return [], []
        else:
            warn(f"EC2 collection error: {e}")
            return [], []
    except EndpointConnectionError:
        warn("Network error when calling EC2 - skipping instances/security groups")
        return [], []

    return instances_out, sgs_out

@retryable
def _s3_list_buckets(s3):
    return s3.list_buckets()

@retryable
def _s3_get_bucket_location(s3, bucket: str):
    return s3.get_bucket_location(Bucket=bucket)

@retryable
def _s3_list_objects_v2(s3, bucket: str, continuation_token: Optional[str] = None, max_keys: int = 1000):
    if continuation_token:
        return s3.list_objects_v2(Bucket=bucket, ContinuationToken=continuation_token, MaxKeys=max_keys)
    else:
        return s3.list_objects_v2(Bucket=bucket, MaxKeys=max_keys)

def _approx_bucket_stats(s3, bucket: str, time_budget_objects: int = 500) -> Tuple[int, int]:
    total = 0
    size = 0
    scanned = 0
    token = None
    while True:
        try:
            page = _s3_list_objects_v2(s3, bucket, token, max_keys=1000)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("AccessDenied", "AccessDeniedException"):
                warn(f"Failed to access S3 bucket '{bucket}': Access Denied")
                return 0, 0
            else:
                warn(f"S3 list error for bucket '{bucket}': {e}")
                return 0, 0

        contents = page.get("Contents", [])
        for obj in contents:
            total += 1
            size += obj.get("Size", 0)
            scanned += 1
            if scanned >= time_budget_objects:
                return total, size

        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return total, size

def collect_s3() -> List[Dict[str, Any]]:
    out = []
    try:
        s3 = boto3.client("s3")
        buckets = _s3_list_buckets(s3).get("Buckets", [])
        for b in buckets:
            name = b.get("Name")
            try:
                loc = _s3_get_bucket_location(s3, name)
                region = loc.get("LocationConstraint") or "us-east-1"
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("AccessDenied", "AccessDeniedException"):
                    warn(f"Failed to access S3 bucket '{name}': Access Denied")
                    region = None
                else:
                    raise

            obj_count, size_bytes = _approx_bucket_stats(s3, name, time_budget_objects=500)
            out.append({
                "bucket_name": name,
                "creation_date": b.get("CreationDate").replace(microsecond=0).isoformat() + "Z" if b.get("CreationDate") else None,
                "region": region,
                "object_count": obj_count,
                "size_bytes": size_bytes
            })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "AccessDeniedException"):
            warn("Access denied for S3 operations - skipping buckets")
            return []
        else:
            warn(f"S3 collection error: {e}")
            return []
    except EndpointConnectionError:
        warn("Network error when calling S3 - skipping buckets")
        return []
    return out


def to_json_output(account_info: Dict[str, Any],
                   iam_users: List[Dict[str, Any]],
                   instances: List[Dict[str, Any]],
                   buckets: List[Dict[str, Any]],
                   sgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    running = sum(1 for i in instances if i.get("state") == "running")
    data = {
        "account_info": account_info,
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": instances,
            "s3_buckets": buckets,
            "security_groups": sgs
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": running,
            "total_buckets": len(buckets),
            "security_groups": len(sgs)
        }
    }
    return data

def _pad(s: Any, w: int) -> str:
    s = "-" if (s is None or s == "") else str(s)
    return (s[:w]).ljust(w)

def print_table(account_info: Dict[str, Any],
                iam_users: List[Dict[str, Any]],
                instances: List[Dict[str, Any]],
                buckets: List[Dict[str, Any]],
                sgs: List[Dict[str, Any]]):
    print(f"AWS Account: {account_info.get('account_id')} ({account_info.get('region') or '-'})")
    print(f"Scan Time: {account_info.get('scan_timestamp')}")
    print()

    
    print(f"IAM USERS ({len(iam_users)} total)")
    print(_pad("Username", 20), _pad("Create Date", 20), _pad("Last Activity", 20), _pad("Policies", 8))
    for u in iam_users:
        print(_pad(u.get("username"), 20),
              _pad((u.get("create_date") or "")[:10], 20),
              _pad((u.get("last_activity") or "")[:10], 20),
              _pad(len(u.get("attached_policies", [])), 8))
    print()

    
    running = sum(1 for i in instances if i.get("state") == "running")
    stopped = sum(1 for i in instances if i.get("state") == "stopped")
    print(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    print(_pad("Instance ID", 20), _pad("Type", 12), _pad("State", 10), _pad("Public IP", 16), _pad("Launch Time", 20))
    for ins in instances:
        print(_pad(ins.get("instance_id"), 20),
              _pad(ins.get("instance_type"), 12),
              _pad(ins.get("state"), 10),
              _pad(ins.get("public_ip") or "-", 16),
              _pad((ins.get("launch_time") or "")[:16], 20))
    print()

    
    print(f"S3 BUCKETS ({len(buckets)} total)")
    print(_pad("Bucket Name", 32), _pad("Region", 12), _pad("Created", 20), _pad("Objects", 10), _pad("Size (MB)", 12))
    for b in buckets:
        mb = round((b.get("size_bytes", 0) or 0) / (1024 * 1024), 1)
        print(_pad(b.get("bucket_name"), 32),
              _pad(b.get("region"), 12),
              _pad((b.get("creation_date") or "")[:10], 20),
              _pad(b.get("object_count"), 10),
              _pad(f"~{mb}", 12))
    print()

    
    print(f"SECURITY GROUPS ({len(sgs)} total)")
    print(_pad("Group ID", 16), _pad("Name", 18), _pad("VPC ID", 16), _pad("Inbound Rules", 13))
    for g in sgs:
        print(_pad(g.get("group_id"), 16),
              _pad(g.get("group_name"), 18),
              _pad(g.get("vpc_id"), 16),
              _pad(len(g.get("inbound_rules", [])), 13))
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None, help="AWS region (for EC2/SG). If omitted, use default.")
    parser.add_argument("--output", type=str, default=None, help="Output file path. If omitted, print to stdout.")
    parser.add_argument("--format", type=str, default="json", choices=["json", "table"], help="Output format.")
    args = parser.parse_args()

    
    try:
        region = validate_region(args.region)
    except ValueError as e:
        error(str(e))
        sys.exit(1)
    except NoRegionError:
        region = None


    try:
        ident = get_identity()
        account_id = ident.get("Account")
        user_arn = ident.get("Arn")
    except NoCredentialsError:
        error("Authentication failed: No AWS credentials found. Use aws configure or env vars.")
        sys.exit(1)
    except ClientError as e:
        error(f"Authentication failed: {e}")
        sys.exit(1)
    except EndpointConnectionError:
        error("Network error during authentication.")
        sys.exit(1)

    account_info = {
        "account_id": account_id,
        "user_arn": user_arn,
        "region": region or os.environ.get("AWS_DEFAULT_REGION"),
        "scan_timestamp": utc_now_iso()
    }

    iam_users = collect_iam_users()
    ec2_instances, sgs = collect_ec2_and_sg(region)
    s3_buckets = collect_s3()

    if args.format == "json":
        payload = to_json_output(account_info, iam_users, ec2_instances, s3_buckets, sgs)
        out = json.dumps(payload, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
        else:
            print(out)
    else:
        if args.output:
            from io import StringIO
            buf = StringIO()
            sys_stdout = sys.stdout
            try:
                sys.stdout = buf
                print_table(account_info, iam_users, ec2_instances, s3_buckets, sgs)
            finally:
                sys.stdout = sys_stdout
            with open(args.output, "w") as f:
                f.write(buf.getvalue())
        else:
            print_table(account_info, iam_users, ec2_instances, s3_buckets, sgs)


if __name__ == "__main__":
    main()
