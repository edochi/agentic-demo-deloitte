terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-04-e58bb0017c73-terraform-state"
    prefix = "prod"
  }
}
