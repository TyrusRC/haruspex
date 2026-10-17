"""Cloud/IaC scanner-finding routing tests (pure Python; no torch/gliner2 needed)."""

import unittest

from haruspex.cloud import CloudResolver, is_cloud_finding
from haruspex.model import Haruspex


class CloudResolverTest(unittest.TestCase):
    def setUp(self):
        self.r = CloudResolver()

    def test_direct_violations_are_real(self):
        for s in ["security group ingress allows 0.0.0.0/0 to port 22",
                  'IAM policy document allows Action "*" on Resource "*"',
                  "S3 Bucket has an ACL defined which allows public READ access",
                  "RDS instance publicly_accessible=true"]:
            self.assertIs(self.r.classify(s), True, s)

    def test_hardened_states_are_safe(self):
        for s in ["aws_s3_bucket_public_access_block sets block_public_acls=true ignore_public_acls=true block_public_policy=true restrict_public_buckets=true",
                  "security group ingress allows TCP 443 from a single corporate CIDR 203.0.113.0/24",
                  "RDS instance publicly_accessible=false and storage_encrypted=true"]:
            self.assertIs(self.r.classify(s), False, s)

    def test_failed_security_check_is_real(self):
        # checkov phrases security checks as recommendations; a FAILED one = violated
        for s in ['Checkov CKV_AWS_111 FAILED: Ensure no IAM policies allow "*" as a statement',
                  "Checkov CKV_AWS_18 FAILED: Ensure there is no open access to back-end resources",
                  "Checkov CKV_AWS_173 FAILED: Check encryption settings for Lambda environment variables"]:
            self.assertIs(self.r.classify(s), True, s)

    def test_ops_nits_are_not_findings(self):
        for s in ["Checkov CKV_AWS_50 FAILED: X-Ray tracing is enabled for Lambda",
                  "Ensure that AWS Lambda function is configured for a Dead Letter Queue",
                  "Ensure API Gateway caching is enabled",
                  "Ensure Create before destroy for API Gateway"]:
            self.assertIs(self.r.classify(s), False, s)

    def test_defers_when_unsure(self):
        self.assertIsNone(self.r.classify("some entirely unrelated sentence about weather"))

    def test_is_cloud_finding(self):
        self.assertTrue(is_cloud_finding({"type": "boolean", "state": "Checkov CKV_AWS_20 failed on aws_s3_bucket"}))
        self.assertTrue(is_cloud_finding({"type": "boolean", "state": "security group allows 0.0.0.0/0"}))
        self.assertFalse(is_cloud_finding({"type": "boolean", "state": "a reflected XSS in the search box"}))


class CloudRoutingTest(unittest.TestCase):
    """decide() must answer a cloud finding from the resolver without touching the model."""

    def test_decide_routes_without_model(self):
        hx = Haruspex(model=None, cloud_resolver=CloudResolver())  # no model loaded
        v = hx.decide({"type": "boolean", "state": "security group ingress allows 0.0.0.0/0 to port 3389"})
        self.assertEqual(v["source"], "cloud_rule")
        self.assertEqual(v["label"], "yes")
        self.assertEqual(v["probability"], 1.0)

        v2 = hx.decide({"type": "boolean",
                        "state": "aws_s3_bucket_public_access_block sets all four settings to true"})
        self.assertEqual(v2["source"], "cloud_rule")
        self.assertEqual(v2["label"], "no")
        self.assertEqual(v2["probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
