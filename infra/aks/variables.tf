variable "subscription_id" {
  description = "Azure subscription to deploy into (Azure for Students). Pass with -var or TF_VAR_subscription_id."
  type        = string
}

variable "location" {
  description = "Azure region. The student subscription's policy allows swedencentral, uaenorth, spaincentral, polandcentral and germanywestcentral."
  type        = string
  default     = "swedencentral"

  validation {
    condition     = contains(["swedencentral", "uaenorth", "spaincentral", "polandcentral", "germanywestcentral"], var.location)
    error_message = "The subscription's policy only allows swedencentral, uaenorth, spaincentral, polandcentral or germanywestcentral."
  }
}

variable "prefix" {
  description = "Name prefix for every resource."
  type        = string
  default     = "reel-history"
}

variable "kubernetes_version" {
  description = "AKS Kubernetes minor version; the latest patch is picked. 1.37 matches the kind cluster."
  type        = string
  default     = "1.37"
}

variable "node_vm_size" {
  description = "VM size of the single node pool. B2s_v2: 2 vCPU, 8 GiB, burstable."
  type        = string
  default     = "Standard_B2s_v2"
}

variable "node_count" {
  description = "Nodes in the pool. 2 x B2s_v2 = 4 vCPUs, inside the subscription's 6-vCPU regional quota."
  type        = number
  default     = 2

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 3
    error_message = "Keep the pool between 1 and 3 nodes (6-vCPU regional quota)."
  }
}

variable "os_disk_size_gb" {
  description = "OS disk per node. 32 GB is plenty for these images and cheaper than the 128 GB default."
  type        = number
  default     = 32
}
