/*--------------------------------------------------------------*/
/*--------------------------------------------------------------*/
/*----------Checkpoint chunk protection code--------------------*/

int enable_chunkprot(int chunkid) { 
	return enabl_chunkprot_using_map(chunkid);
}

size_t nv_disablprot(void *addr, int *curr_chunkid) {
	int chunkid = 0;
	size_t length = 0;

	length =remove_chunk_prot(addr, &chunkid);
	*curr_chunkid = chunkid;
#ifdef _RMT_PRECOPY
	//    set_chkpt_type(ASYNC_LCL_CKPT);
	//   send_lock_avbl_sig(SIGUSR1);
#endif
#ifdef _USE_FAULT_PATTERNS

	int nxt_chunk=0;
	//if(find_chunk_fault_list(chunkid)){	
	if(!chunk_fault_lst_freeze)
		add_chunk_fault_lst(chunkid);

	else {
		/*get the next chunk to be protected
		set the protection of the chunk*/
		nxt_chunk = get_next_chunk(chunkid);
		if(!nxt_chunk)
			goto end;

		if(!nxt_chunk || enable_chunkprot(nxt_chunk))
			if(prev_proc_id == 1)
				fprintf(stderr,"chun protection failed\n");
	}
#endif
	end:
	return length;
}


int set_chunkprot() { 
	return set_chunkprot_using_map();
}



int add_to_fault_lst(int id) {

	int val = 0;

	fault_chunk[id] = 0;
	val = fault_chunk[id];
	val++;
	fault_chunk[id] = val;	

	if(!stop_history_coll && (checpt_cnt > 0))
		fault_hist[id]++;
	return 0;
}

int clear_fault_lst() {

	int faultid = 0;
	for( fault_itr= fault_chunk.begin();
			fault_itr!=fault_chunk.end();
			++fault_itr){

		faultid = (*fault_itr).first; 
		fault_chunk[faultid]=0;
	}
	return 0;
}

#ifdef _USE_CHECKPOINT
#ifdef ENABLE_CHECKPOINT


int reg_for_signal(int procid) {

	/*termination signal*/
	app_exec_finish_sig(procid, SIGUSR2);

	return register_ckpt_lock_sig(procid, SIGUSR1);
}


int init_checkpoint(int procid) {

	init_shm_lock(procid);

	//pthread_mutex_init(&chkpt_mutex, NULL);
	//pthread_mutex_lock(&chkpt_mutex);
	//mutex_set = 1;
	return 0;
}


int  chkpt_all_chunks(rbtree_node n, int *cmt_chunks) {

	int ret =-1;
	chunkobj_s *chunkobj = NULL;
	char gen_key[256];
	ULONG lat_ns, cycles;


#ifdef VALIDATE_CHKSM
	long hash;
#endif

	if (n == NULL) {
		return 0;
	}

	if (n->right != NULL) {
		ret = chkpt_all_chunks(n->right,cmt_chunks);
	}

	chunkobj = (chunkobj_s *)n->value;

	if(chunkobj) {

#ifdef _ASYNC_LCL_CHK
		if(chunkobj->dirty) {
#endif
			void *src = chunkobj->log_ptr;
			void *dest = chunkobj->nv_ptr;

			assert(src);
			assert(dest);
			assert(chunkobj->length);

			//#ifdef _NVDEBUG
			if(prev_proc_id == 1)
				fprintf(stdout,"commiting chk no:%d chunk %u "
						"and size %u \t"
						"committed? %d \n",
						local_chkpt_cnt,
						chunkobj->chunkid,
						chunkobj->length,
						chunkobj->dirty);
			//#endif
			*cmt_chunks = *cmt_chunks + 1;

			//if(prev_proc_id == 1)
			//print_stats(prev_proc_id);

#ifdef _ASYNC_LCL_CHK
			chunkobj->dirty = 0;
#endif

#ifdef _ASYNC_RMT_CHKPT
			chunkobj->rmt_nvdirtchunk = 1;
#endif

#ifdef _COMPARE_PAGES
			compare_content_hash(src,dest, chunkobj->length);
#endif
			size_t cmpr_len = 0;
			//if(prev_proc_id == 1)
			memcpy_delay(dest,src,chunkobj->length);
			//snappy::RawCompress((const char *)src, chunkobj->length, (char *)dest, &cmpr_len);
			//fprintf(stdout,"Before %u After Compre %u \n", chunkobj->length, cmpr_len);

#ifdef _NVSTATS
			proc_stat.tot_cmtdata += chunkobj->length;
			add_to_chunk_memcpy(chunkobj);
#endif

#ifdef VALIDATE_CHKSM
			bzero(gen_key, 256);
			sha1_mykeygen(src, gen_key,
					CHKSUM_LEN, 16, chunkobj->length);

			hash = gen_id_from_str(gen_key);

			chunkobj->checksum = hash;
#endif
			ret = 0;

#ifdef _ASYNC_LCL_CHK
		}
#endif

	}

	if (n->left != NULL) {
		return chkpt_all_chunks(n->left,cmt_chunks);
	}
	return ret;
}

int  chkpt_all_vmas(rbtree_node n) {

	int ret =-1;
	mmapobj_s *t_mmapobj = NULL;
	rbtree_node root;
	int cmt_chunks =0, tot_chunks=0;

	//assert(n);
	if (n == NULL) {
		return 0;
	}

	if (n->right != NULL) {
		ret = chkpt_all_vmas(n->right);
	}

	t_mmapobj = (mmapobj_s *)n->value;

	if(t_mmapobj) {
		if(t_mmapobj->chunkobj_tree) {

#ifdef _NVDEBUG
			print_mmapobj(t_mmapobj);
#endif
			root = t_mmapobj->chunkobj_tree->root;
			if(root)

#ifdef _USE_GPU_CHECKPT
				ret = gpu_chkpt_all_chunks(root, &cmt_chunks);
#else
			ret = chkpt_all_chunks(root, &cmt_chunks);
#endif
		}
	}

	if (n->left != NULL) {
		return chkpt_all_vmas(n->left);
	}

	tot_chunks = get_chnk_cnt_frm_map();

#ifdef _NVDEBUG
	fprintf(stdout,"total chunks %d, cmt chunks %d\n",
			tot_chunks, cmt_chunks);
#endif

	return ret;
}


int nv_chkpt_all(rqst_s *rqst, int remoteckpt) {

	int process_id = -1;
	proc_s *proc_obj= NULL;
	rbtree_node root;
	int ret = 0;

	local_chkpt_cnt++;

#ifdef _NVSTATS
	long cmttime =0;
	struct timeval ckpt_start_time, ckpt_end_time;

	gettimeofday(&ckpt_start_time, NULL);
#endif

	if(!rqst)
		goto error;

	process_id = rqst->pid;
	proc_obj= find_proc_obj(process_id);

#ifdef _NVDEBUG
	fprintf(stdout,"invoking commit "
			"for process %d \n",
			rqst->pid);

	assert(proc_obj);
	assert(proc_obj->mmapobj_tree);
	assert(proc_obj->mmapobj_tree->root);
#endif
	if(!proc_obj)
		goto error;

	if(!(proc_obj->mmapobj_tree))
		goto error;

	root = proc_obj->mmapobj_tree->root;
	if(!root)
		goto error;

#ifdef _NVSTATS
	gettimeofday(&commit_end, NULL);
	cmttime = simulation_time( commit_start,commit_end);

	chkpt_itr_time = cmttime;

	add_stats_commit_freq(process_id, cmttime);
#endif

	//set_acquire_chkpt_lock(process_id);
	ret = chkpt_all_vmas(root);
#ifdef _ASYNC_RMT_CHKPT
	set_ckptdirtflg(process_id);
#endif
	//if(prev_proc_id == 1)
	//print_stats(process_id);

	//disable_chkpt_lock(process_id);
	//
	incr_lcl_chkpt_cnt();

#ifdef _COMPARE_PAGES
	hash_clear();
#endif

#ifdef _FAULT_STATS
	if(prev_proc_id == 1)
		set_chunkprot();
#endif //_FAULT_STATS


#ifdef _ASYNC_LCL_CHK
	set_chunkprot();
	/*if(checpt_cnt ==1)
		stop_history_coll = 1;
	clear_fault_lst();
	checpt_cnt++;*/
#endif

	error:

#ifdef _ASYNC_RMT_CHKPT
	set_chkpt_type(SYNC_LCL_CKPT);
	send_lock_avbl_sig(SIGUSR1);
#else
	pthread_cond_signal(&dataPresentCondition);
	//pthread_mutex_unlock(&chkpt_mutex);
#endif


#ifdef _USE_FAULT_PATTERNS
	if(!check_chunk_fault_lst_empty())
		chunk_fault_lst_freeze =1;
	//if(prev_proc_id == 1)
	//	print_chunk_fault_lst();
#endif


#ifdef _NVSTATS
	gettimeofday(&ckpt_end_time, NULL);

	add_stats_chkpt_time(process_id,
			simulation_time(ckpt_start_time,ckpt_end_time));

	gettimeofday(&commit_start, NULL);
#endif

	if(!ret){
#ifdef _NVDEBUG
		printf("nv_chkpt_all: succeeded for procid"
				" %d \n",proc_obj->pid);
#endif
		return ret;
	}

#ifdef _NVDEBUG
	printf("nv_chkpt_all: failed for procid"
			" %d \n",proc_obj->pid);
#endif
	return -1;
}
#endif




int start_asyn_lcl_chkpt(int chunkid) {

	int faultid=0;
	int fault_cnt =0;


	//fprintf(stdout,"proc id %d \n",prev_proc_id);
	for( fault_itr= fault_chunk.begin(); 
			fault_itr!=fault_chunk.end(); 
			++fault_itr){

		faultid = (*fault_itr).first;			
		fault_cnt = (*fault_itr).second;
		//if(prev_proc_id == 1){
		// fprintf(stdout,"chunk id %d fault_hist[id] %d \n",faultid, fault_hist[faultid]);
		//}
		/*if(fault_hist[faultid]  &&(fault_cnt >= (fault_hist[faultid]))){
			copy_dirty_chunk(faultid, 1);
		}else{
			copy_dirty_chunk(faultid, 0);	
		}*/
		copy_dirty_chunk(faultid, 1);	
#ifdef _RMT_PRECOPY
		set_chkpt_type(ASYNC_LCL_CKPT);
		send_lock_avbl_sig(SIGUSR1);
#endif

	}


#ifdef _ASYNC_RMT_CHKPT
	set_ckptdirtflg(0);


#endif
	return 0;
}

int  proc_rmt_load_data(int procid) {

	proc_s *proc_obj;
	proc_obj  = find_process(procid);
	int perm = 0;
	if(!proc_obj) {
		proc_obj = load_process(procid, perm);
		if(!proc_obj){
			goto error;
		}else{
			return SUCCESS;
		}
	}
	return EXISTS;
	error:
	return FAILURE;
}


//checkpoint related code
void* proc_rmt_chkpt(int procid, size_t *bytes, int check_dirtypgs,
		int num_procs, int myrank) {

	void *ret = NULL;
	int perm = 0; 
	int lddata =-1;
	int ckpttype =-1;

#ifdef _ASYNC_RMT_CHKPT
	wait_for_signal:
	lddata= wait_for_chkpt_sig(procid, SIGUSR1);

	ckpttype = get_chkpt_type();

	if(lddata == NO_SHM_AVBL){
		//fprintf(stdout,"waiting for signal \n");
		goto wait_for_signal;
	}
	if(lddata == INCORR_INTERVAL){
		/*check if remote addrspace exists*/
		if(proc_rmt_load_data(procid) == SUCCESS){

#ifdef _ARMCI_CHECKPOINT
			rmtchkpt_setup_memory(num_procs,myrank);
#endif
		}
		lddata = -1;
		//fprintf(stdout,"incorrect interval "
		//"but remote addr space can be created\n");
		goto wait_for_signal;
	}

#else
	pthread_mutex_lock(&chkpt_mutex);
#endif

#ifndef _ASYNC_RMT_CHKPT	

#ifdef _NVDEBUG
	fprintf(stdout, "waitig for dirty flag set\n");
#endif

	while (!get_dirtyflag(procid)) {
		pthread_cond_wait(&dataPresentCondition, &chkpt_mutex);
	}

#ifdef _NVDEBUG
	fprintf(stdout, "dirty flag set\n");
#endif

#else //_ASYNC_RMT_CHKPT

#ifdef _NVDEBUG
	fprintf(stdout, "before acquire_chkpt_lock\n");
#endif

	if(acquire_chkpt_lock(procid)){
		fprintf(stdout,"ckpt failed for %d \n",procid);
		return NULL;
	}

#ifdef _NVDEBUG
	fprintf(stdout, "after acquire_chkpt_lock "
			"waiting for dirtly flag\n");
#endif
	/*while (!get_dirtyflag(procid)) {
		sleep(1);
	}*/
#endif

	lddata = proc_rmt_load_data(procid);
	if(lddata == FAILURE){
		fprintf(stdout,"proc_rmt_chkpt: reading proc"
				"%d info failed \n", procid);
		goto exit;
	}

	/*we are creating for first time*/
	if( lddata == SUCCESS) { 
		/*create remote address space*/
#ifdef _ARMCI_CHECKPOINT
		rmtchkpt_setup_memory(num_procs,myrank);
#endif
	}

#ifdef _NVDEBUG
	fprintf(stdout, "copying checkpoint\n");
#endif

	ret = copy_proc_checkpoint(procid, bytes, 
			check_dirtypgs,
			ckpttype);
#ifdef _NVSTATS
	//if(!(local_chkpt_cnt % 2))
	print_rmt_chkpt_stats(procid, myrank);
#endif


#ifdef _NVDEBUG
	fprintf(stdout, "all checkpoint data ready\n");
#endif

	disable_ckptdirtflag(procid);

	exit:


#ifdef _ASYNC_RMT_CHKPT
	disable_chkpt_lock(procid);
#else
	pthread_mutex_unlock(&chkpt_mutex);
#endif

	return ret;
}


#ifdef _ARMCI_CHECKPOINT
int rmtchkpt_setup_memory(int nranks, int myrank){

	return setup_memory(nranks, myrank);
}

int nv_register_chkpt_peer(int peer){

	register_chkpt_peer(peer);
}
#endif

long get_chkpt_itr_time() {

	return chkpt_itr_time;
}


#ifdef _USE_SHMFORPCM

void* create_shm(size_t shmsz) 
{
	int segid;
	void *shm;
	struct shmid_ds shmbuffer;
	size_t segsz;
	int key = 5678;

	segid = shmget(key, 1024*1024*500, IPC_CREAT | 0666);

	if ((shm = shmat(segid, NULL, 0)) == (char *) -1) {
		perror("shmat");
		exit(1);
	}

	return shm;
}
#endif


void* disk_mmap(size_t size,  nvarg_s *s, size_t offset, int create_map) {

	char file_name[256];
	bzero(file_name,256);
	proc_s *proc_obj = NULL;
	int fdesc = -1;
	ULONG strt_addr = NULL;
	ULONG ret_addr = NULL;
	int pid = 0;
	int fd =-1;
	mmapobj_s *mmap_obj = NULL;

	//request id cannot be zero
	assert(s->proc_id);
	pid = s->proc_id;
	proc_obj = find_proc_obj(pid);

	//we havent seen any reqst from proc
	if(!proc_obj) {

		fd = setup_map_file(file_name,size+1);
		if (fd == -1) {
			perror("file open error\n");
			return NULL;
		}
#ifdef _USE_SHMFORPCM
		strt_addr = (ULONG) create_shm(MAX_MMAP_SIZE);
#else 

#ifdef _USE_PCM_ANONMAP
		strt_addr = (ULONG)mmap(0,MAX_MMAP_SIZE,
				PROT_NV_RW, MAP_ANONYMOUS|MAP_PRIVATE,-1, 0);
#else
		strt_addr = (ULONG)mmap(0,size,
				PROT_NV_RW, MAP_SHARED, fd, 0);

#endif //_USE_PCM_ANONMAP
#endif //_USE_SHMFORPCM
		assert(strt_addr);

#ifdef _NVSTATS
		add_stats_mmap(s->proc_id, MAX_MMAP_SIZE);
#endif
		insert_mmapobj_node((ULONG)strt_addr, size,s->vma_id,s->proc_id);

#ifdef _USE_CHECKPOINT
		record_vmas(s->vma_id, MAX_MMAP_SIZE);
		//record_vma_ghash(s->vma_id, MAX_MMAP_SIZE);
#endif
		return (void *)strt_addr;
	}else {
		mmap_obj = find_mmapobj(s->vma_id,proc_obj);
		assert(mmap_obj);
		strt_addr = mmap_obj->data_addr;
		assert(strt_addr);
		ret_addr = strt_addr + offset;
#ifdef _NVDEBUG		
		fprintf(stdout,"disk map offset %u\n",offset);
#endif
		assert(ret_addr);
		return (void *)ret_addr;		
	}
}






#ifdef _USE_DISKMAP


int disk_flush(int pid ) {

	proc_s *proc_obj = NULL;
	mmapobj_s *mmap_obj = NULL;
	unsigned long strt_addr =0;

	proc_obj = find_proc_obj(pid);
	assert(proc_obj);

	mmap_obj = find_mmapobj(DATA_VMAID,proc_obj);
	assert(mmap_obj);

	strt_addr = mmap_obj->data_addr;
	assert(strt_addr);

	fprintf(stdout,"flush mmap_obj->length %u %lu \n", mmap_obj->length, strt_addr);
	assert(msync((void *)strt_addr,  mmap_obj->length, MS_SYNC) >= 0);
	assert( (int )munmap ((void *)strt_addr, mmap_obj->length ) >= 0);
	assert(disk_fd >= 0);
	close(disk_fd);

	return 0;
}

void* _disk_map(rqst_s *rqst) {

	nvarg_s a;
	proc_s *proc_obj;
	mmapobj_s *mmap_obj;
	size_t bytes =0;
	size_t offset =0;
	void* ret = NULL;
	//if a diskmap does not exist
	//create it
	int create_if_not_exist = 1;

	a.fd = -1;
	//same vmaid
	a.vma_id = DATA_VMAID;
	a.pflags = 1;
	a.noPersist = 0;
	a.proc_id = rqst->pid;
	bytes = rqst->bytes;

	proc_obj = find_proc_obj(a.proc_id);
	if(!proc_obj) {
		offset = 0;
		goto callmap;
	}

	mmap_obj = find_mmapobj(a.vma_id, proc_obj);
	if(!mmap_obj) {
		//ideally we should have 
		//found it
		assert(0);
	}
	offset = mmap_obj->offset;

	callmap:
	ret = disk_mmap(bytes,&a, offset,create_if_not_exist); 
	assert(ret);


	return  ret;
}



#endif

/*---------------------------------------------------------------------------*/
/*----------------GPU Checkpoint related code--------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef _USE_GPU_CHECKPT

int  gpu_chkpt_all_chunks(rbtree_node n, int *cmt_chunks) {

	int ret =-1;
	chunkobj_s *chunkobj = NULL;
	char gen_key[256];
	ULONG lat_ns, cycles;

	if (n == NULL) {
		return 0;
	}
	if (n->right != NULL) {
		ret = gpu_chkpt_all_chunks(n->right,cmt_chunks);
	}
	chunkobj = (chunkobj_s *)n->value;

	if(chunkobj) {
		if(!chunkobj->chunk_commit) {

			void *src = chunkobj->log_ptr;
			void *dest = chunkobj->nv_ptr;
			assert(src);
			assert(dest);
			assert(chunkobj->length);

#ifdef _NVDEBUG
			*cmt_chunks = *cmt_chunks + 1;
#endif
			memcpy_delay(dest,src,chunkobj->length);

#ifdef _NVSTATS
			proc_stat.tot_cmtdata += chunkobj->length;
			add_to_chunk_memcpy(chunkobj);
#endif			
			ret = 0;
		}else {
			chunkobj->chunk_commit = 0;
		}

	}
	if (n->left != NULL) {
		return gpu_chkpt_all_chunks(n->left,cmt_chunks);
	}
	retint nvchkptvec(void *ipvec) {

		unsigned long start, end;
		std::vector<unsigned long> *vec;

		vec = (std::vector<unsigned long> *)ipvec;

		while (!vec->empty()){

			chunkobj_s *chunk = NULL;
			void *addr =(void *)vec->back();
			vec->pop_back();
			assert(addr);

			//find the chunk corresponding to addr
			chunk = (chunkobj_s *)get_chunk_from_map_o1(addr);
			if(!chunk) {
				chunk = (chunkobj_s *)get_chunk_from_map(addr);
				if(!chunk){
					assert(0);
				}
			}
			void *src = chunk->log_ptr;
			void *dest = chunk->nv_ptr;
			assert(src);
			assert(dest);
			assert(chunk->length);
			memcpy_delay(dest, src, chunk->length);
			chunk->chunk_commit = 1;
		}
		return 0;
	}
#endifurn ret;
}


int nvchkptvec(void *ipvec) {

	unsigned long start, end;
	std::vector<unsigned long> *vec;

	vec = (std::vector<unsigned long> *)ipvec;

	while (!vec->empty()){

		chunkobj_s *chunk = NULL;
		void *addr =(void *)vec->back();
		vec->pop_back();
		assert(addr);

		//find the chunk corresponding to addr
		chunk = (chunkobj_s *)get_chunk_from_map_o1(addr);
		if(!chunk) {
			chunk = (chunkobj_s *)get_chunk_from_map(addr);
			if(!chunk){
				assert(0);
			}
		}
		void *src = chunk->log_ptr;
		void *dest = chunk->nv_ptr;
		assert(src);
		assert(dest);
		assert(chunk->length);
		memcpy_delay(dest, src, chunk->length);
		chunk->chunk_commit = 1;
	}
	return 0;
}
#endif

#endif

