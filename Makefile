update.sh:
	gcloud components update && gcloud components install beta

confirm_quota.sh: update.sh
	gcloud beta compute regions describe asia-east1

start_gpu: confirm_quota.sh
	gcloud beta compute instances create gpu-deep-learner --machine-type n1-standard-2 --zone asia-east1-a --accelerator type=nvidia-tesla-k80,count=1 --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud --boot-disk-size 100GB --maintenance-policy TERMINATE --restart-on-failure
	gcloud compute ssh gpu-deep-learner --zone asia-east1-a

delete_gpu:
	gcloud compute instances delete gpu-deep-learner --delete-disks=all --zone=asia-east1-a
