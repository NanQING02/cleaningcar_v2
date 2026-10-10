/* 只拦截经核验gst-rockchip的fd NV12->BGR任务；没有新RGA操作或CPU转换。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <rga/RgaApi.h>
#include <rga/rga.h>

_Static_assert(sizeof(rga_info_t) == 696, "rga_info_t ABI size changed");
_Static_assert(offsetof(rga_info_t, core) == 188, "rga_info_t core moved");
_Static_assert(RGA3_SCHEDULER_CORE0 == 1 && RGA3_SCHEDULER_CORE1 == 2, "core mask changed");

typedef int (*blit_fn)(rga_info_t *, rga_info_t *, rga_info_t *);
static blit_fn real_blit;
static struct stat plugin_stat;
static _Atomic unsigned long jobs, guarded, rejected, foreign, failed;
static _Thread_local void *cached_base;
static _Thread_local int cached_match;

static void fatal(const char *message) {
    dprintf(STDERR_FILENO, "[rga3-guard] 初始化失败：%s\n", message);
    _Exit(78);
}

static int same_file(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino;
}

__attribute__((constructor)) static void initialize(void) {
    const char *active = getenv("CLEANINGCAR_RGA3_ACTIVE");
    const char *lib = getenv("CLEANINGCAR_RGA3_LIB");
    const char *plugin = getenv("CLEANINGCAR_RGA3_PLUGIN");
    const char *nonce = getenv("CLEANINGCAR_RGA3_NONCE");
    if (!active || strcmp(active, "1") || !nonce || !*nonce || !lib || !plugin)
        fatal("缺少已核验的启动上下文");
    struct stat lib_stat, symbol_stat;
    if (stat(lib, &lib_stat) || stat(plugin, &plugin_stat)) fatal("系统库路径不可访问");
    void *handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!handle) fatal("系统librga无法加载");
    real_blit = (blit_fn)dlsym(handle, "c_RkRgaBlit");
    Dl_info symbol;
    if (!real_blit || !dladdr((void *)real_blit, &symbol) || !symbol.dli_fname ||
        stat(symbol.dli_fname, &symbol_stat) || !same_file(&lib_stat, &symbol_stat))
        fatal("真实c_RkRgaBlit来源不匹配");
    /* 本灰度启动器拒绝其他LD_PRELOAD；避免ffmpeg/上传/守护子进程再加载本库。 */
    if (setenv("CLEANINGCAR_RGA3_READY", nonce, 1)) fatal("初始化凭据写入失败");
    unsetenv("LD_PRELOAD");
    dprintf(STDERR_FILENO, "[rga3-guard] ready pid=%ld abi=696/188 mask=3 scope=gst-rockchip fd-NV12-to-BGR\n", (long)getpid());
}

static int plugin_caller(void *return_address) {
    Dl_info info;
    if (!dladdr(return_address, &info) || !info.dli_fbase || !info.dli_fname) return 0;
    if (cached_base != info.dli_fbase) {
        struct stat caller;
        cached_base = info.dli_fbase;
        cached_match = !stat(info.dli_fname, &caller) && same_file(&caller, &plugin_stat);
    }
    return cached_match;
}

static int fd_buffer(const rga_info_t *info) {
    return info && info->fd >= 0 && !info->virAddr && !info->phyAddr && !info->hnd && !info->handle;
}

static int known_conversion(const rga_info_t *src, const rga_info_t *dst, const rga_info_t *pat) {
    if (pat || !fd_buffer(src) || !fd_buffer(dst)) return 0;
    if (src->rect.format != RK_FORMAT_YCbCr_420_SP || dst->rect.format != RK_FORMAT_BGR_888) return 0;
    if (src->rotation || dst->rotation || src->blend || dst->blend || src->colorkey_en || dst->colorkey_en) return 0;
    if (src->rect.width != dst->rect.width || src->rect.height != dst->rect.height) return 0;
    if (src->rect.xoffset || src->rect.yoffset || dst->rect.xoffset || dst->rect.yoffset) return 0;
    if ((src->rd_mode != 0 && src->rd_mode != 1) || (dst->rd_mode != 0 && dst->rd_mode != 1)) return 0;
    if (src->rect.width < 2 || src->rect.height < 2 || src->rect.width > 4096 || src->rect.height > 4096) return 0;
    if (src->rect.wstride < src->rect.width || src->rect.hstride < src->rect.height ||
        dst->rect.wstride < dst->rect.width || dst->rect.hstride < dst->rect.height) return 0;
    if (src->rect.wstride % 16 || dst->rect.wstride % 16) return 0;
    return 1;
}

static void summary(const char *phase) {
    dprintf(STDERR_FILENO, "[rga3-guard] %s pid=%ld calls=%lu guarded=%lu rejected=%lu foreign=%lu failed=%lu\n",
        phase, (long)getpid(), atomic_load(&jobs), atomic_load(&guarded), atomic_load(&rejected), atomic_load(&foreign), atomic_load(&failed));
}

int c_RkRgaBlit(rga_info_t *src, rga_info_t *dst, rga_info_t *pat) {
    unsigned long n = atomic_fetch_add(&jobs, 1) + 1;
    if (!plugin_caller(__builtin_return_address(0))) {
        atomic_fetch_add(&foreign, 1);
        return real_blit(src, dst, pat);
    }
    if (!known_conversion(src, dst, pat)) {
        unsigned long count = atomic_fetch_add(&rejected, 1) + 1;
        if (count == 1 || count % 4096 == 0) summary("unsupported-job");
        /* 未验证任务直接失败，绝不悄悄回退RGA2或虚拟地址路径。 */
        errno = EINVAL;
        return -EINVAL;
    }
    int src_core = src->core, dst_core = dst->core;
    src->core = dst->core = RGA3_SCHEDULER_CORE0 | RGA3_SCHEDULER_CORE1;
    atomic_fetch_add(&guarded, 1);
    int result = real_blit(src, dst, pat);
    int saved_errno = errno;
    src->core = src_core;
    dst->core = dst_core;
    if (result != 0) atomic_fetch_add(&failed, 1);
    if (n == 1 || n % 4096 == 0) summary("stats");
    errno = saved_errno;
    return result;
}

int cleaningcar_rga3_guard_api_version(void) { return 1; }
__attribute__((destructor)) static void finish(void) { summary("exit"); }
