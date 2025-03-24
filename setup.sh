#!/usr/bin/env bash

export OWNER=arshdeep
export AWS_REGION=us-west-2
export CLUSTER_NAME=emr-eks-ml
export S3_BUCKET_NAME=${CLUSTER_NAME}-${AWS_REGION}

mkdir -p config && mkdir -p manifests

echo "create cluster and node group config files..."
cat << EOF > ./manifests/cluster.yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  tags:
    Owner: ${OWNER}
EOF

cat << EOF > ./manifests/nodegroup.yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  tags:
    Owner: ${OWNER}

managedNodeGroups:
- name: nodegroup
  desiredCapacity: 2
  instanceType: m5.xlarge
EOF

cat << EOF > ./manifests/nodegroup-spot.yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  tags:
    Owner: ${OWNER}

managedNodeGroups:
- name: nodegroup-spot
  desiredCapacity: 3
  instanceTypes:
  - m5.xlarge
  - m5a.xlarge
  - m4.xlarge
  spot: true
EOF

echo "create pod templates..."
cat << EOF > ./config/driver_pod_template.yml
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    eks.amazonaws.com/capacityType: ON_DEMAND
EOF

cat << EOF > ./config/executor_pod_template.yml
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    eks.amazonaws.com/capacityType: SPOT
EOF