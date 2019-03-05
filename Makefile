OPENMP = 1
zfp-0.5.3/lib/libzfp.so: zfp-0.5.3.tar.gz
	tar -xzvf zfp-0.5.3.tar.gz
	cd zfp-0.5.3 && make ZFP_WITH_OPENMP=$(OPENMP)

zfp-0.5.3.tar.gz: 
	wget https://computation.llnl.gov/projects/floating-point-compression/download/zfp-0.5.3.tar.gz

clean:
	rm -rf zfp-0.5.3*
