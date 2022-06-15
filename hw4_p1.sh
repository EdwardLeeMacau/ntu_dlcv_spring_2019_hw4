# TODO: create shell script for Problem 1
if ! [ -f "./problem1.pth" ]; then
	wget -O ./problem1.pth https://www.dropbox.com/s/03345sy74ir3s25/problem1.pth?dl=0
fi

python3 predict_cnn.py --resume ./problem1.pth --video $1 --label $2 --output $3
