#!/bin/bash

cat > run.sh <<EOF
#!/bin/bash
time ~/gpu_flow/compute_flow -g 0 -vp "../videos_downsample_cut/\$1/" -op "../videos_flow/\$1/"
EOF

chmod +x run.sh

for type in all1 all2 esea1 esea2 ; do
  ./run.sh "$type"
done
