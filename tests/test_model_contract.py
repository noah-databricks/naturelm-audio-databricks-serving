import unittest
from pathlib import Path


class ModelContractTests(unittest.TestCase):
    def test_model_from_code_contract_is_present(self):
        source = Path("src/model/naturelm_pyfunc.py").read_text(encoding="utf-8")
        self.assertIn("mlflow.models.set_model(NatureLMAudioModel())", source)
        self.assertIn('"audio_base64"', source)
        self.assertIn('"query"', source)
        self.assertIn("local_files_only=True", source)
        self.assertIn("device=self.device", source)

    def test_endpoint_deployment_is_explicit_and_scale_to_zero(self):
        source = Path("src/notebooks/02_deploy_and_validate.py").read_text(encoding="utf-8")
        self.assertIn('cfg["confirm_deploy"] != "YES"', source)
        self.assertIn('"scale_to_zero_enabled": True', source)
        self.assertIn('"workload_type": cfg["workload_type"]', source)
        self.assertIn('deployment_action = "unchanged"', source)
        self.assertIn('deployment_action = "update_in_progress"', source)


if __name__ == "__main__":
    unittest.main()
