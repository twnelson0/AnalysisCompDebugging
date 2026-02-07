#!/usr/bin/env bash

cat <<EOF > shell
#!/usr/bin/env bash


if [ "\$1" == "" ]; then
	export COFFEA_IMAGE=coffeateam/coffea-dask-cc7:latest-py3.10
else
	export COFFEA_IMAGE=\$1
fi

EXTERNAL_BIND=\${PWD} singularity exec -B \${PWD}:/srv -B /etc/condor -B /scratch -B /hdfs -B /nfs_scratch -B /var/condor --env=_CONDOR_SPOOL=/var/condor/spool --pwd /srv \\
	/cvmfs/unpacked.cern.ch/registry.hub.docker.com/\${COFFEA_IMAGE}  \\
	/bin/bash --rcfile /srv/.bashrc
EOF

chmod u+x shell .bashrc
echo "Wrote shell and .bashrc to current directory. You can delete this file. Run ./shell to start the singularity shell"
