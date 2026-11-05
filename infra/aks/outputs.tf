output "resource_group" {
  description = "Resource group holding everything; deleting it removes the whole deployment."
  value       = azurerm_resource_group.aks.name
}

output "cluster_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "acr_name" {
  value = azurerm_container_registry.acr.name
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "image_repository" {
  description = "Pass to helm as --set image.repository=..."
  value       = "${azurerm_container_registry.acr.login_server}/reel-history"
}

output "get_credentials_command" {
  description = "Merges the cluster into ~/.kube/config and switches kubectl to it."
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.aks.name} --name ${azurerm_kubernetes_cluster.aks.name} --overwrite-existing"
}
