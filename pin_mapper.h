/*
 * pin_mapper.h
 *
 *  Created on: Apr 19, 2012
 *      Author: hendrix
 */

#ifndef PIN_MAPPER_H_
#define PIN_MAPPER_H_

#define PIN_SHM_KEY 5678
#define PIN_SHM_SZ 1048576

#ifdef __cplusplus
extern "C" {
#endif

void* CreateMapFile(char *filepath, unsigned long bytes);
char* Writeline(unsigned long strt, unsigned long end);
char *CreateSharedMem();

#ifdef __cplusplus
}
#endif


#endif /* PIN_MAPPER_H_ */
