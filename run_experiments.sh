#!/bin/bash
# Script to reproduce results
for ((i=0;i<5;i+=1))
do
    python es_grad.py \
	--env "HalfCheetah-v2" \
	--seed $i \
	--use_td3 \
	--output "HalfCheetah-v2"

	python es_grad.py \
	--env "Hopper-v2" \
	--seed $i \
	--use_td3 \
	--output "Hopper-v2"

	python es_grad.py \
	--env "Walker2d-v2" \
	--seed $i \
	--use_td3 \
	--output "Walker2d-v2"

	python es_grad.py \
	--env "Ant-v2" \
	--seed $i \
	--use_td3 \
	--output "Ant-v2"

	python es_grad.py \
	--env "Swimmer-v2" \
	--seed $i \
	--use_td3 \
	--output "Swimmer-v2"
done
