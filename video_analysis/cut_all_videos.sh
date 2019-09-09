#!/bin/bash -e


type=all
dur=20

cat > run.sh <<EOF
#!/bin/bash
time time ffmpeg -y -ss 0 -i "../videos_downsample/$type/300_10/\$1"  -y -ss 0 -t $dur "../videos_downsample_cut/$type/\$1"
EOF

chmod +x run.sh

cat ../videos_downsample/$type.txt | parallel -j 8 ./run.sh ::::

type=esea
dur=60

cat > run.sh <<EOF
#!/bin/bash
time time ffmpeg -y -ss 0 -i "../videos_downsample/$type/300_10/\$1"  -y -ss 0 -t $dur "../videos_downsample_cut/$type/\$1"
EOF

chmod +x run.sh

cat ../videos_downsample/$type.txt | parallel -j 8 ./run.sh ::::
