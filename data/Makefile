makefolders:
	mkdir ~/data
	mkdir ~/data/train
	mkdir ~/data/validation 
	mkdir ~/data/train/cats 
	mkdir ~/data/train/dogs
	mkdir ~/data/validation/cats
	mkdir ~/data/validation/dogs

download_files: makefolders
	gsutil -m cp -r gs://kagglecatdog/train/cat*jpg ~/data/train/cats/
	gsutil -m cp -r gs://kagglecatdog/train/dog*jpg ~/data/train/dogs/

split_data:
	shuf -zen2500 ~/data/train/cats/*jpg | xargs -0 mv -t ~/data/validation/cats/
	shuf -zen2500 ~/data/train/dogs/*jpg | xargs -0 mv -t ~/data/validation/dogs/	
