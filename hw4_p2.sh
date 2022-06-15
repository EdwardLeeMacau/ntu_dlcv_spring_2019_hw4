# TODO: create shell script for Problem 2
if ! [ -f "./problem2.pth" ]; then
	wget -O ./problem2.pth https://www.dropbox.com/s/e2t4ngf4vdac76k/problem2.pth?dl=0
fi

python3 predict_rnn.py --resume ./problem2.pth --video $1 --label $2 --output $3
