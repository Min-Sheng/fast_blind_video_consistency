CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky1/image1_HR_crop/ \
--stylePath images/mystyle/1.jpg \
--outf wct_sky1_style1 \
--fineSize 1024

CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky1/image1_HR_crop/ \
--stylePath images/mystyle/2.jpg \
--outf wct_sky1_style2 \
--fineSize 1024

CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky1/image1_HR_crop/ \
--stylePath images/mystyle/3.jpg \
--outf wct_sky1_style3 \
--fineSize 1024

CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky2/image2_HR_crop/ \
--stylePath images/mystyle/1.jpg \
--outf wct_sky2_style1 \
--fineSize 1024

CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky2/image2_HR_crop/ \
--stylePath images/mystyle/2.jpg \
--outf wct_sky2_style2 \
--fineSize 1024

CUDA_VISIBLE_DEVICES=2 python3 WCT.py --cuda \
--contentPath ../content/sky2/image2_HR_crop/ \
--stylePath images/mystyle/3.jpg \
--outf wct_sky2_style3 \
--fineSize 1024

