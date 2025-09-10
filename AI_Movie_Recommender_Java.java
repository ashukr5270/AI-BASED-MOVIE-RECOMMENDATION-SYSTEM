import java.util.*;
import java.util.stream.Collectors;

public class AI_Movie_Recommender_Java {

    // Simple Movie representation
    static class Movie {
        final int id;
        final String title;
        final String description;
        final Set<String> genres;

        Movie(int id, String title, String description, String... genres) {
            this.id = id;
            this.title = title;
            this.description = description == null ? "" : description.toLowerCase();
            this.genres = Arrays.stream(genres).map(String::toLowerCase).collect(Collectors.toSet());
        }

        public String toString() {
            return id + ": " + title;
        }
    }

    // Simple User representation storing ratings (1.0-5.0)
    static class User {
        final int id;
        final Map<Integer, Double> ratings = new HashMap<>();

        User(int id) { this.id = id; }

        void rate(int movieId, double rating) { ratings.put(movieId, rating); }
    }

    // Recommender implementation
    static class Recommender {
        final List<Movie> movies;
        final Map<Integer, Movie> movieById;
        final List<User> users;

        // Precomputed content vectors (term -> weight)
        final Map<Integer, Map<String, Double>> contentVectors = new HashMap<>();

        // Precomputed item-item similarities for collaborative filtering
        final Map<Integer, Map<Integer, Double>> itemSims = new HashMap<>();

        Recommender(List<Movie> movies, List<User> users) {
            this.movies = movies;
            this.users = users;
            this.movieById = new HashMap<>();
            for (Movie m : movies) movieById.put(m.id, m);

            buildContentVectors();
            buildItemSimilarities();
        }

        // ---------------- Content-based (TF-IDF + cosine) ----------------
        void buildContentVectors() {
            // Build doc-term frequencies
            List<Map<String, Integer>> docTermCounts = new ArrayList<>();
            Map<String, Integer> df = new HashMap<>();

            for (Movie m : movies) {
                Map<String, Integer> termCount = new HashMap<>();
                List<String> tokens = tokenize(m.description);
                // include genres as tokens too
                tokens.addAll(m.genres);
                Set<String> seen = new HashSet<>();
                for (String t : tokens) {
                    termCount.put(t, termCount.getOrDefault(t, 0) + 1);
                    if (seen.add(t)) df.put(t, df.getOrDefault(t, 0) + 1);
                }
                docTermCounts.add(termCount);
            }

            int N = movies.size();
            for (int i = 0; i < movies.size(); i++) {
                Map<String, Integer> tf = docTermCounts.get(i);
                Map<String, Double> vector = new HashMap<>();
                double norm = 0.0;
                for (Map.Entry<String, Integer> e : tf.entrySet()) {
                    String term = e.getKey();
                    int termFreq = e.getValue();
                    double idf = Math.log((N + 1.0) / (1 + df.getOrDefault(term, 0))) + 1.0; // smoothed idf
                    double weight = termFreq * idf;
                    vector.put(term, weight);
                    norm += weight * weight;
                }
                norm = Math.sqrt(norm);
                // normalize
                if (norm > 0) {
                    for (Map.Entry<String, Double> e : new ArrayList<>(vector.entrySet())) {
                        vector.put(e.getKey(), e.getValue() / norm);
                    }
                }
                contentVectors.put(movies.get(i).id, vector);
            }
        }

        List<String> tokenize(String text) {
            if (text == null) return new ArrayList<>();
            String cleaned = text.replaceAll("[^a-z0-9 ]", " ").toLowerCase();
            String[] parts = cleaned.split("\\s+");
            List<String> out = new ArrayList<>();
            for (String p : parts) if (!p.isEmpty() && p.length() > 1) out.add(p);
            return out;
        }

        static double cosine(Map<String, Double> a, Map<String, Double> b) {
            if (a == null || b == null) return 0.0;
            // iterate over smaller map
            Map<String, Double> small = a.size() <= b.size() ? a : b;
            Map<String, Double> large = a.size() > b.size() ? a : b;
            double dot = 0.0;
            for (Map.Entry<String, Double> e : small.entrySet()) {
                double bv = large.getOrDefault(e.getKey(), 0.0);
                dot += e.getValue() * bv;
            }
            return dot; // both vectors normalized already
        }

        List<Movie> recommendContent(int userId, int k) {
            User user = users.stream().filter(u -> u.id == userId).findFirst().orElse(null);
            if (user == null) return Collections.emptyList();

            // Build user profile vector: weighted sum of liked movies' content vectors
            Map<String, Double> profile = new HashMap<>();
            double normFactor = 0.0;
            for (Map.Entry<Integer, Double> entry : user.ratings.entrySet()) {
                int mid = entry.getKey();
                double rating = entry.getValue();
                Map<String, Double> mv = contentVectors.get(mid);
                if (mv == null) continue;
                for (Map.Entry<String, Double> t : mv.entrySet()) {
                    profile.put(t.getKey(), profile.getOrDefault(t.getKey(), 0.0) + t.getValue() * (rating - 3.0));
                    // center ratings around neutral 3.0 to highlight positive/negative
                }
                normFactor += Math.abs(rating - 3.0);
            }
            // normalize profile
            if (normFactor > 0) {
                for (Map.Entry<String, Double> e : new ArrayList<>(profile.entrySet()))
                    profile.put(e.getKey(), e.getValue() / normFactor);
            }

            // score each unseen movie
            PriorityQueue<Map.Entry<Movie, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(Map.Entry::getValue));
            for (Movie m : movies) {
                if (user.ratings.containsKey(m.id)) continue; // skip seen
                double score = cosine(profile, contentVectors.get(m.id));
                pq.offer(new AbstractMap.SimpleEntry<>(m, score));
                if (pq.size() > k) pq.poll();
            }
            List<Movie> out = new ArrayList<>();
            while (!pq.isEmpty()) out.add(0, pq.poll().getKey());
            return out;
        }

        // ---------------- Collaborative Filtering (item-based) ----------------
        void buildItemSimilarities() {
            // Build item-user rating map
            Map<Integer, Map<Integer, Double>> itemRatings = new HashMap<>(); // movieId -> userId->rating
            for (User u : users) {
                for (Map.Entry<Integer, Double> e : u.ratings.entrySet()) {
                    itemRatings.computeIfAbsent(e.getKey(), x -> new HashMap<>()).put(u.id, e.getValue());
                }
            }

            for (Movie m1 : movies) {
                Map<Integer, Double> sims = new HashMap<>();
                Map<Integer, Double> r1 = itemRatings.getOrDefault(m1.id, Collections.emptyMap());
                for (Movie m2 : movies) {
                    if (m1.id == m2.id) continue;
                    Map<Integer, Double> r2 = itemRatings.getOrDefault(m2.id, Collections.emptyMap());
                    double sim = cosineRatings(r1, r2);
                    sims.put(m2.id, sim);
                }
                itemSims.put(m1.id, sims);
            }
        }

        double cosineRatings(Map<Integer, Double> a, Map<Integer, Double> b) {
            // center each user's rating by avg user rating if desired - here simple raw cosine
            if (a.isEmpty() || b.isEmpty()) return 0.0;
            // iterate over smaller
            Map<Integer, Double> small = a.size() <= b.size() ? a : b;
            Map<Integer, Double> large = a.size() > b.size() ? a : b;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (Map.Entry<Integer, Double> e : small.entrySet()) {
                double aval = e.getValue();
                double bval = large.getOrDefault(e.getKey(), 0.0);
                dot += aval * bval;
            }
            for (double v : a.values()) na += v * v;
            for (double v : b.values()) nb += v * v;
            if (na == 0 || nb == 0) return 0.0;
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        }

        List<Movie> recommendCollaborative(int userId, int k) {
            User user = users.stream().filter(u -> u.id == userId).findFirst().orElse(null);
            if (user == null) return Collections.emptyList();

            Map<Integer, Double> scores = new HashMap<>();
            Map<Integer, Double> weightSum = new HashMap<>();

            for (Map.Entry<Integer, Double> e : user.ratings.entrySet()) {
                int seenMid = e.getKey();
                double rating = e.getValue();
                Map<Integer, Double> sims = itemSims.getOrDefault(seenMid, Collections.emptyMap());
                for (Map.Entry<Integer, Double> s : sims.entrySet()) {
                    int candidate = s.getKey();
                    if (user.ratings.containsKey(candidate)) continue;
                    double sim = s.getValue();
                    scores.put(candidate, scores.getOrDefault(candidate, 0.0) + sim * rating);
                    weightSum.put(candidate, weightSum.getOrDefault(candidate, 0.0) + Math.abs(sim));
                }
            }

            Map<Integer, Double> predicted = new HashMap<>();
            for (Map.Entry<Integer, Double> e : scores.entrySet()) {
                double denom = weightSum.getOrDefault(e.getKey(), 1.0);
                predicted.put(e.getKey(), e.getValue() / denom);
            }

            PriorityQueue<Map.Entry<Integer, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(Map.Entry::getValue));
            for (Map.Entry<Integer, Double> e : predicted.entrySet()) {
                pq.offer(e);
                if (pq.size() > k) pq.poll();
            }
            List<Movie> out = new ArrayList<>();
            while (!pq.isEmpty()) out.add(0, movieById.get(pq.poll().getKey()));
            return out;
        }

        // ---------------- Hybrid ----------------
        List<Movie> recommendHybrid(int userId, int k, double alphaContentWeight) {
            // alphaContentWeight in [0,1] weight given to content-based; remaining to collaborative
            User user = users.stream().filter(u -> u.id == userId).findFirst().orElse(null);
            if (user == null) return Collections.emptyList();

            List<Movie> contentList = recommendContent(userId, Math.max(50, k * 3));
            List<Movie> collabList = recommendCollaborative(userId, Math.max(50, k * 3));

            Map<Integer, Double> combined = new HashMap<>();
            // score content: normalized rank-based score
            for (int i = 0; i < contentList.size(); i++) {
                combined.put(contentList.get(i).id, combined.getOrDefault(contentList.get(i).id, 0.0) + (contentList.size() - i) * alphaContentWeight);
            }
            for (int i = 0; i < collabList.size(); i++) {
                combined.put(collabList.get(i).id, combined.getOrDefault(collabList.get(i).id, 0.0) + (collabList.size() - i) * (1 - alphaContentWeight));
            }

            PriorityQueue<Map.Entry<Integer, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(Map.Entry::getValue));
            for (Map.Entry<Integer, Double> e : combined.entrySet()) {
                pq.offer(e);
                if (pq.size() > k) pq.poll();
            }
            List<Movie> out = new ArrayList<>();
            while (!pq.isEmpty()) out.add(0, movieById.get(pq.poll().getKey()));
            return out;
        }

    }

    // ---------------- Demo main ----------------
    public static void main(String[] args) {
        List<Movie> movies = Arrays.asList(
            new Movie(1, "The Space Between Stars", "An astronaut struggles with loneliness while exploring distant galaxies. Dramatic sci-fi about isolation and discovery.", "Sci-Fi", "Drama"),
            new Movie(2, "Romantic Rhapsody", "A young musician falls in love and fights for her big break in a bustling city. Heartfelt romance with music.", "Romance", "Music"),
            new Movie(3, "Mystery Manor", "Detectives investigate strange occurrences at a Victorian manor. A twisting whodunit with dark secrets.", "Mystery", "Thriller"),
            new Movie(4, "Galactic Battles", "An interstellar war unfolds between rival fleets. Action-packed space opera with epic battles.", "Action", "Sci-Fi"),
            new Movie(5, "City of Laughter", "A group of comedians try to save their favorite club from closing. A feel-good comedy about friendship and stand-up.", "Comedy"),
            new Movie(6, "Secrets of the Mind", "A psychological thriller exploring memory and identity after a traumatic event.", "Thriller", "Drama")
        );

        User u1 = new User(101);
        u1.rate(1, 5.0); // loved space drama
        u1.rate(4, 4.0); // liked space action

        User u2 = new User(102);
        u2.rate(2, 5.0);
        u2.rate(5, 4.0);

        User u3 = new User(103);
        u3.rate(3, 5.0);
        u3.rate(6, 4.5);

        List<User> users = Arrays.asList(u1, u2, u3);

        Recommender rec = new Recommender(movies, users);

        int targetUser = 101;
        System.out.println("Content-based recommendations for user " + targetUser + ":");
        for (Movie m : rec.recommendContent(targetUser, 3)) System.out.println(" - " + m);

        System.out.println("\nCollaborative recommendations for user " + targetUser + ":");
        for (Movie m : rec.recommendCollaborative(targetUser, 3)) System.out.println(" - " + m);

        System.out.println("\nHybrid recommendations (0.6 content weight) for user " + targetUser + ":");
        for (Movie m : rec.recommendHybrid(targetUser, 5, 0.6)) System.out.println(" - " + m);

        System.out.println("\nTip: Replace demo data with CSV-loaded datasets and tune weighting.");
    }
}