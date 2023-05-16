std::priority_queue<std::pair<double, Node *> >
search_layer_new(Node *q, Node *ep, int ef, int lc) {
    double d = distance(&(ep->data), &(q->data), lc);
    std::unordered_set<Node *> v{ep};                          // set of visited elements
    std::priority_queue<std::pair<double, Node *> > candidates; // set of candidates
    std::priority_queue<std::pair<double, Node *> > w;          // dynamic list of found nearest neighbors
    candidates.emplace(-d, ep);
    w.emplace(d, ep);

    while (!candidates.empty()) {
        Node *c = candidates.top().second; // extract nearest element from c to q
        double c_dist = candidates.top().first;
        candidates.pop();
        Node *f = w.top().second; // get furthest element from w to q
        double f_dist = w.top().first;
        if (-c_dist > f_dist) {
            break;
        }
        for (Node *e: c->neighbors[lc]) {
            if (v.find(e) == v.end()) {
                v.emplace(e);
                e->parent = c;   // record parent
                f_dist = w.top().first;
                double distance_e_q = distance(&(e->data), &(q->data), lc);
                if (distance_e_q < f_dist || w.size() < ef) {
                    candidates.emplace(-distance_e_q, e);
                    w.emplace(distance_e_q, e);
                    if (w.size() > ef) {
                        w.pop();
                    }
                }
            }
        }
    }

    std::priority_queue<std::pair<double, Node *> > min_w;
    while (!w.empty()) {
        min_w.emplace(-w.top().first, w.top().second);
        w.pop();
    }
    return min_w;
}

// activation function sigmoid
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}