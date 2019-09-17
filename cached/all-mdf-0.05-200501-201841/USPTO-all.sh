#!/usr/bin/env bash
python pygrams.py -ts -ei gradients -nts 5 -mpq 50 -sma kalman -dt 2018/05/31 -tsdf 2012/06/01 -tsdt 2016/06/01 --test -pns 1 2 3 4 5 6 -dh publication_date -ds USPTO-granted-lite-all.pkl.bz2
