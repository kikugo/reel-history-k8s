# AKS for Reel History: one resource group holding an AKS cluster (free control plane, one small
# node pool) and a container registry the cluster can pull from. Destroy the same day.

locals {
  tags = {
    project = "reel-history"
    phase   = "8"
  }
}

# Everything lives in one group, so `terraform destroy` (or deleting the group) removes it all.
resource "azurerm_resource_group" "aks" {
  name     = "rg-${var.prefix}-aks"
  location = var.location
  tags     = local.tags
}

# ACR names are global and alphanumeric only, so add a random suffix.
resource "random_string" "acr_suffix" {
  length  = 6
  upper   = false
  special = false
}

# Basic is the cheapest tier and enough for three images. No admin user: AKS pulls with its
# managed identity (role assignment below) and images are built with `az acr build`.
resource "azurerm_container_registry" "acr" {
  name                = "reelhistory${random_string.acr_suffix.result}"
  resource_group_name = azurerm_resource_group.aks.name
  location            = azurerm_resource_group.aks.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = local.tags
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "aks-${var.prefix}"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  dns_prefix          = var.prefix
  kubernetes_version  = var.kubernetes_version
  # Free tier: no charge for the control plane, no uptime SLA. Fine for a few hours of demo.
  sku_tier = "Free"

  # One system pool runs both AKS's own pods and the app; a separate user pool would need more
  # vCPUs than the 6-vCPU quota leaves.
  default_node_pool {
    name                        = "system"
    vm_size                     = var.node_vm_size
    node_count                  = var.node_count
    os_disk_type                = "Managed"
    os_disk_size_gb             = var.os_disk_size_gb
    temporary_name_for_rotation = "rotate"
    tags                        = local.tags
  }

  # Required in azurerm v5. Manual: the node pool above is the only one; no auto-provisioned pools.
  node_provisioning_profile {
    mode = "Manual"
  }

  # The cluster's own identity (for load balancers, disks); AKS also creates a separate kubelet
  # identity for the nodes, which gets AcrPull below.
  identity {
    type = "SystemAssigned"
  }

  # Azure CNI in overlay mode: pods get IPs from a private overlay range, not the VNet, so no
  # subnet sizing. A Standard load balancer fronts the Envoy Gateway service.
  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    load_balancer_sku   = "standard"
  }

  tags = local.tags
}

# Let the nodes pull images from this registry, and only this registry.
resource "azurerm_role_assignment" "kubelet_acr_pull" {
  scope                            = azurerm_container_registry.acr.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}
