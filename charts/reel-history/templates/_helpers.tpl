{{/* Labels shared by every object of one service. Usage: include "rh.labels" (list $ "recommender") */}}
{{- define "rh.labels" -}}
{{- $root := index . 0 -}}
app.kubernetes.io/name: {{ index . 1 }}
app.kubernetes.io/part-of: reel-history
app.kubernetes.io/instance: {{ $root.Release.Name }}
app.kubernetes.io/version: {{ $root.Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ $root.Release.Service }}
helm.sh/chart: {{ printf "%s-%s" $root.Chart.Name $root.Chart.Version }}
{{- end }}

{{/* Selector labels: must never change after install, so no version in here. */}}
{{- define "rh.selector" -}}
app.kubernetes.io/name: {{ index . 1 }}
app.kubernetes.io/instance: {{ (index . 0).Release.Name }}
{{- end }}

{{- define "rh.image" -}}
{{- $root := index . 0 -}}
{{ printf "%s/%s:%s" $root.Values.image.repository (index . 1) $root.Values.image.tag }}
{{- end }}

{{- define "rh.containerSecurity" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end }}
