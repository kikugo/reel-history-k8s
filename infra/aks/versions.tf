terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.12"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }
}

provider "azurerm" {
  features {}
  # Explicit, so a stray `az account set` can never point an apply at another subscription.
  subscription_id = var.subscription_id
  # v5 registers no resource providers by default; Compute, Network, ContainerService and
  # ContainerRegistry were registered once by hand (see DECISIONS.md, Phase 8).
}
