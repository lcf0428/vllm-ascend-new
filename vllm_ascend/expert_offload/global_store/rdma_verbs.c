/* Optional Host-to-Host RC RDMA READ backend. No NPU runtime dependency.
 * Build: cc -shared -fPIC -O2 rdma_verbs.c -libverbs -o libexpert_rdma.so
 * TCP carries QP metadata only. Expert bytes move through IBV_WR_RDMA_READ.
 */
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct endpoint {
    uint64_t address, size;
    uint32_t rkey, qpn, psn;
    uint16_t lid, mtu;
    uint8_t gid[16];
};

struct connection {
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    int port, gid_index;
};

void expert_rdma_close(struct connection *c) {
    if (!c) return;
    if (c->qp) ibv_destroy_qp(c->qp);
    if (c->mr) ibv_dereg_mr(c->mr);
    if (c->cq) ibv_destroy_cq(c->cq);
    if (c->pd) ibv_dealloc_pd(c->pd);
    if (c->context) ibv_close_device(c->context);
    free(c);
}

struct connection *expert_rdma_open(const char *device, int port, int gid_index,
                                   void *address, uint64_t size, struct endpoint *out) {
    int count = 0;
    struct ibv_device **devices = ibv_get_device_list(&count);
    struct connection *c = calloc(1, sizeof(*c));
    struct ibv_port_attr pa;
    union ibv_gid gid;
    if (!c || !devices) goto fail;
    for (int i = 0; i < count; ++i) {
        if (!device[0] || !strcmp(ibv_get_device_name(devices[i]), device)) {
            c->context = ibv_open_device(devices[i]);
            break;
        }
    }
    ibv_free_device_list(devices);
    devices = NULL;
    if (!c->context || ibv_query_port(c->context, port, &pa) ||
        pa.state != IBV_PORT_ACTIVE || ibv_query_gid(c->context, port, gid_index, &gid)) goto fail;
    c->pd = ibv_alloc_pd(c->context);
    if (!c->pd) goto fail;
    c->mr = ibv_reg_mr(c->pd, address, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    c->cq = ibv_create_cq(c->context, 8, NULL, NULL, 0);
    if (!c->mr || !c->cq) goto fail;
    struct ibv_qp_init_attr qa = {0};
    qa.send_cq = qa.recv_cq = c->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = qa.cap.max_recv_wr = 2;
    qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qa);
    if (!c->qp) goto fail;
    struct ibv_qp_attr init = {0};
    init.qp_state = IBV_QPS_INIT;
    init.port_num = port;
    init.pkey_index = 0;
    init.qp_access_flags = IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(c->qp, &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) goto fail;
    c->port = port;
    c->gid_index = gid_index;
    memset(out, 0, sizeof(*out));
    out->address = (uintptr_t)address;
    out->size = size;
    out->rkey = c->mr->rkey;
    out->qpn = c->qp->qp_num;
    out->psn = 0;
    out->lid = pa.lid;
    out->mtu = pa.active_mtu;
    memcpy(out->gid, gid.raw, 16);
    return c;
fail:
    if (devices) ibv_free_device_list(devices);
    expert_rdma_close(c);
    return NULL;
}

int expert_rdma_connect(struct connection *c, const struct endpoint *peer,
                         const struct endpoint *local) {
    struct ibv_qp_attr a = {0};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = peer->mtu < local->mtu ? peer->mtu : local->mtu;
    a.dest_qp_num = peer->qpn;
    a.rq_psn = peer->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.dlid = peer->lid;
    a.ah_attr.port_num = c->port;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    a.ah_attr.grh.sgid_index = c->gid_index;
    a.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                      IBV_QP_MIN_RNR_TIMER)) return -1;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = local->psn;
    a.max_rd_atomic = 1;
    return ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int expert_rdma_read(struct connection *c, const struct endpoint *peer,
                     uint64_t offset, uint32_t size, double timeout) {
    if (!size || size > c->mr->length || offset > peer->size || size > peer->size - offset) return -1;
    struct ibv_sge sge = {(uintptr_t)c->mr->addr, size, c->mr->lkey};
    struct ibv_send_wr wr = {0}, *bad = NULL;
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = peer->address + offset;
    wr.wr.rdma.rkey = peer->rkey;
    if (ibv_post_send(c->qp, &wr, &bad)) return -2;
    double deadline = monotonic_seconds() + timeout;
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(c->cq, 1, &wc);
        if (n < 0) return -3;
        if (n) return wc.status == IBV_WC_SUCCESS ? 0 : -4;
        if (monotonic_seconds() >= deadline) {
            struct ibv_qp_attr err = {.qp_state = IBV_QPS_ERR};
            ibv_modify_qp(c->qp, &err, IBV_QP_STATE);
            return -5; /* Registration remains live until QP destruction. */
        }
        usleep(50);
    }
}
