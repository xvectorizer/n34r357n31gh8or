std::priority_queue <std::pair<double, Node *>>
search_layer_new(Node *q, Node *ep, int ef, int lc, bool is_log = false) {
    float x = 0.8;
    std::mt19937 rng;
    std::uniform_real_distribution<float> rand_0_1(0.000001, 0.999999);
    if (is_log) std::fstream log_file_new("log_search_new.txt", std::ios_base::app);
    if (is_log) log_file_new << q->id << " " << lc << " | " << ep->id << " ";
    double d = distance(&(ep->data), &(q->data));
    std::unordered_set < Node * > v{ep};                                   // set of visited elements
    std::priority_queue <std::pair<double, Node *>> candidates;
    std::vector < Node * > candidates_vec;
    std::priority_queue <std::pair<double, Node *>> w;                  // dynamic list of found nearest neighbors
    candidates.emplace(-d, ep);
    w.emplace(d, ep);
    candidates_vec.push_back(ep);
    ep->next = 0;

    while (!candidates.empty() && !candidates_vec.empty()) {
        Node *c = nullptr;
        double c_dist;
        if (unif_0_1(rng) < x) {
            c = candidates.top().second;
            c_dist = candidates.top().first;
            if (c->next >= c->neighbors[lc].size()) {
                candidates.pop();
                continue;
            }
        } else {
            // random choose an element from candidates_vec
            int index = floor(rand_0_1(rng) * candidates_vec.size());
            c = candidates_vec[index];
            c_dist = 0;
            if (c->next >= c->neighbors[lc].size()) {
                candidates_vec[index] = candidates_vec[candidates_vec.size() - 1];
                candidates_vec.pop_back();
                continue;
            }
        }

        Node *f = w.top().second; // get furthest element from w to q
        double f_dist = w.top().first;
        if (-c_dist > f_dist) {
            break;
        }

        // if (c->next >= c->neighbors[lc].size()) {
        //     candidates.pop();
        // } else {
        Node *e = c->neighbors[lc][c->next];
        c->next++;
        if (v.find(e) == v.end()) {
            v.emplace(e);
            e->parent = c; // record parent
            f = w.top().second;
            double distance_e_q = distance(&(e->data), &(q->data));
            if (distance_e_q < f_dist || w.size() < ef) {
                e->next = 0;
                candidates.emplace(-distance_e_q, e);
                candidates_vec.push_back(e);
                // candidates_distance.emplace(-distance_e_q, e);
                log_file_new << e->id << " ";
                w.emplace(distance_e_q, e);
                if (w.size() > ef) {
                    w.pop();
                }
            }
        }

    }
    log_file_new << std::endl;
    std::priority_queue <std::pair<double, Node *>> min_w;
    while (!w.empty()) {
        min_w.emplace(-w.top().first, w.top().second);
        w.pop();
    }
    return min_w;
}