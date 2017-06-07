# vppext

An extension of library vpp

A fast embedded implementation of a line detection algorithm using a dense one-to-one Hough transform combined with an efficient clustering method for detecting in a very fast way the dominant lines in the image.

Design a line tracking method, to temporally associate every line to its successive positions along the video. The basic idea is to perform point-based tracking in the Hough parameter space.

Reconstruct a rough 3d structure of the environment (in the context of indoor mobile robotics), using the position and nature (vertical, perspective) of the lines, as well as their motions.

Moving lines https://www.dropbox.com/s/u5xmiqcu0xii98g/mooving.avi?dl=0

output with hough https://www.dropbox.com/s/0jm57w49esuj979/boule.avi?dl=0

Tracking 1 https://www.dropbox.com/s/o4lhpa4hczvs9dc/onThebouleImage.avi?dl=0

Tracking 2 https://www.dropbox.com/s/epwdbok7rbhfm8s/withHough.avi?dl=0
