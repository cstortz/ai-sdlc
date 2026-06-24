{{/*
Expand the name of the chart.
*/}}
{{- define "ai-sdlc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ai-sdlc.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ai-sdlc.labels" -}}
helm.sh/chart: {{ include "ai-sdlc.name" . }}-{{ .Chart.Version }}
app.kubernetes.io/name: {{ include "ai-sdlc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Postgres internal service hostname
*/}}
{{- define "ai-sdlc.postgresHost" -}}
{{- printf "%s-postgres-postgresql" .Release.Name }}
{{- end }}

{{/*
Redis internal service hostname
*/}}
{{- define "ai-sdlc.redisHost" -}}
{{- printf "%s-redis-master" .Release.Name }}
{{- end }}

{{/*
Neo4j bolt URI
*/}}
{{- define "ai-sdlc.neo4jBoltUri" -}}
{{- printf "bolt://%s-neo4j-neo4j:7687" .Release.Name }}
{{- end }}
