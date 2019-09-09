#!/bin/bash -e

type=esea
dur=60
jpgq=60

cat > run.sh <<EOF
#!/bin/bash
time python -u ./extract_image_frames.py --src_dir ../videos_downsample/$type/300_10/ --dst_dir ../videos_frames/$type --file_name "\$1" --max_dur $dur --jpg_quality $jpgq
EOF

chmod +x run.sh

cat ../videos_downsample/$type.txt | parallel -j 4 ./run.sh ::::



type=all
dur=20

cat > run.sh <<EOF
#!/bin/bash
time python -u ./extract_image_frames.py --src_dir ../videos_downsample/$type/300_10/ --dst_dir ../videos_frames/$type --file_name "\$1" --max_dur $dur --jpg_quality $jpgq
EOF

chmod +x run.sh

cat ../videos_downsample/$type.txt | parallel -j 4 ./run.sh ::::

