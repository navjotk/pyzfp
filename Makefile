OPENMP = 0
zfp-0.5.3/lib/libzfp.so: zfp-0.5.3
	cd zfp-0.5.3 && make ZFP_WITH_OPENMP=$(OPENMP)

zfp-0.5.3: 
	git clone -b release0.5.3 https://github.com/LLNL/zfp.git zfp-0.5.3

clean:
	rm -rf zfp-0.5.3*
