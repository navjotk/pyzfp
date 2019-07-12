OPENMP = 0
zfp-0.5.5/lib/libzfp.so: zfp-0.5.5
	cd zfp-0.5.5 && make ZFP_WITH_OPENMP=$(OPENMP)

zfp-0.5.5: 
	git clone -b release0.5.5 https://github.com/LLNL/zfp.git zfp-0.5.5

clean:
	rm -rf zfp-0.5.5
