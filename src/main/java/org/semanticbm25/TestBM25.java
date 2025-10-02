package org.semanticbm25;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.json.JSONArray;
import org.json.JSONObject;
import org.jsoup.Jsoup;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * This class creates tables in the Postgresql database and tests the BM25
 * (You must create an empty database and fill in the URL_DATABASE, USER, PASSWORD parameters).
 * The createBM25Database method creates tables in the database based on the parameters maxKNN, minSim, lsim and the word2vec model.
 * If you change the parameters maxKNN, minSim, lsim, then you need to clear the database and re-execute createBM25Database.
 * You can repeat the test without creating the tables again. To do this, comment out the line createBM25Database(word2Vec, c); in the main method.
 *
 * word2VecModelPath - the path to the file with the word2vec model.
 * fileFolderPath - the path to the folder with the tokenized documents.
 * testQuestionFile - the path to the JSON file with the questions.
 * maxKNN - This is a parameter that indicates how many semantically closest words need to be extracted from the model.
 * minSim - The minimum semantic value at which a word will be considered close in context.
 * lsim - The l parameter for calculating semantic TF.
 * TOP_k - Limits the number of documents in the response.
 * BM25_k and BM25_b - BM25 Parameters.
 *
 * @author Aleksei Shrank
 * @version 1.0
 * @since 2025-09-01
 */

public class TestBM25 {
    static String URL_DATABASE = "jdbc:postgresql://localhost:5432/scopusTestTokenNF";
    static String USER = "postgres";
    static String PASSWORD = "postgres";
    static String word2VecModelPath = "\\BEIR NFCorpus\\model300Token.bin";
    static String fileFolderPath = "\\BEIR NFCorpus\\DocToken";
    static String testQuestionFile = "\\BEIR NFCorpus\\PrepQueriesBM25RAW.json";

    static int maxKNN = 100; //number of K semantics nearest
    static double minSim = 0.6; //minimum level of semantics
    static double lsim = 0.8;
    static int TOP_k = 10000000;
    static double BM25_k = 1.7;
    static double BM25_b = 0.75;

    public static void main(String[] args) {
        try {
            Connection c = getConnection();
            Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(word2VecModelPath);
            createBM25Database(word2Vec, c);
            testBM25(c);
        } catch (SQLException e) {
            System.err.println("SQLException error: " + e.getMessage());
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static void createBM25Database(Word2Vec word2Vec, Connection c) throws SQLException {
        String sqlWords = "CREATE TABLE words (" +
                "word INTEGER PRIMARY KEY," +
                "idf DOUBLE PRECISION," +
                "idfsem DOUBLE PRECISION" +
                ")";

        String sqlDocuments = "CREATE TABLE documents (" +
                "id TEXT PRIMARY KEY," +
                "doclen INTEGER" +
                ")";

        String sqlTf = "CREATE TABLE tf (" +
                "id TEXT," +
                "word INTEGER," +
                "tf INTEGER," +
                "PRIMARY KEY (id, word)," +
                "FOREIGN KEY (id) REFERENCES documents(id) ON DELETE CASCADE," +
                "FOREIGN KEY (word) REFERENCES words(word) ON DELETE CASCADE" +
                ")";
        Statement stmt = c.createStatement();
        stmt.executeUpdate(sqlWords);
        stmt.executeUpdate(sqlDocuments);
        stmt.executeUpdate(sqlTf);

        PreparedStatement pstmtWord = c.prepareStatement("INSERT INTO words (word) VALUES (?) ");
        Set<Integer> allWords = new HashSet<>(word2Vec.getVocab().words().stream().filter(word -> word.length() < 30).map(Integer::parseInt).collect(Collectors.toSet()));
        ProgressBar pb = new ProgressBarBuilder()
                .setTaskName("Words")
                .setInitialMax(allWords.size())
                .setStyle(ProgressBarStyle.ASCII)
                .build();
        for (int word : allWords) {
            pb.step();
            pstmtWord.setInt(1, word);
            pstmtWord.addBatch();
        }
        pb.close();
        pstmtWord.executeLargeBatch();

        Pattern WORD_PATTERN = Pattern.compile("\\d+");
        PreparedStatement pstmt = c.prepareStatement("INSERT INTO documents (id, doclen) VALUES (?, ?) " +
                "ON CONFLICT (id) DO UPDATE SET doclen = EXCLUDED.doclen");

        PreparedStatement pstmtTF = c.prepareStatement("INSERT INTO tf (id, word, tf) VALUES (?, ?, ?) " +
                "ON CONFLICT (id, word) DO UPDATE SET tf = EXCLUDED.tf");

        File[] files = new File(fileFolderPath).listFiles((dir, name) -> name.endsWith(".txt"));
        ProgressBar pbi = new ProgressBarBuilder()
                .setTaskName("Files")
                .setInitialMax(files.length)
                .setStyle(ProgressBarStyle.ASCII)
                .build();
        for (File file : files) {
            String docId = file.toPath().getFileName().toString().replaceFirst("(?i)\\.txt$", "");
            String content = null;
            int wordCount = 0;
            try {
                content = Files.readString(file.toPath());
                Matcher matcher = WORD_PATTERN.matcher(content);
                Map<Integer, Integer> freq = new HashMap<>();
                while (matcher.find()) {
                    wordCount++;
                    int w = Integer.parseInt(matcher.group().toLowerCase());
                    freq.put(w, freq.getOrDefault(w, 0) + 1);
                }
                pstmt.setString(1, docId);
                pstmt.setInt(2, wordCount);
                pstmt.executeUpdate();

                freq.forEach((word, count) -> {
                    try {
                        if (allWords.contains(word)) {
                            pstmtTF.setString(1, docId);
                            pstmtTF.setInt(2, word);
                            pstmtTF.setInt(3, count);
                            pstmtTF.addBatch();
                        }
                    } catch (SQLException e) {
                        throw new RuntimeException(e);
                    }
                });
                pstmtTF.executeBatch();
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (SQLException e) {
                throw new RuntimeException(e);
            }
            pbi.step();
        }
        pbi.close();

        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tf_word ON tf(word)");
        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tf_id ON tf(id)");
        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tf_id_word ON tf(id, word)");

        String sqlIDF = "WITH document_count AS (\n" +
                "    SELECT COUNT(*) as total FROM documents\n" +
                "),\n" +
                "word_doc_counts AS (\n" +
                "    SELECT word, COUNT(DISTINCT id) as doc_count\n" +
                "    FROM tf\n" +
                "    GROUP BY word\n" +
                ")\n" +
                "UPDATE words\n" +
                "SET idf = LN(\n" +
                "    ((d.total - COALESCE(wdc.doc_count, 0) + 0.5) /\n" +
                "    (COALESCE(wdc.doc_count, 0) + 0.5)) + 1\n" +
                ")\n" +
                "FROM document_count d, word_doc_counts wdc\n" +
                "WHERE words.word = wdc.word;";
        stmt.executeUpdate(sqlIDF);


        String sqlTfSem = "CREATE TABLE tfsem (" +
                "id TEXT," +
                "word INTEGER," +
                "tfsem DOUBLE PRECISION," +
                "PRIMARY KEY (id, word)," +
                "FOREIGN KEY (id) REFERENCES documents(id) ON DELETE CASCADE," +
                "FOREIGN KEY (word) REFERENCES words(word) ON DELETE CASCADE" +
                ")";
        stmt.executeUpdate(sqlTfSem);
        PreparedStatement pstmtTFSem = c.prepareStatement("INSERT INTO tfsem (id, word, tfsem) VALUES (?, ?, ?) " +
                "ON CONFLICT (id, word) DO UPDATE SET tfsem = EXCLUDED.tfsem");

        ProgressBar pbis = new ProgressBarBuilder()
                .setTaskName("TFsem")
                .setInitialMax(allWords.size())
                .setStyle(ProgressBarStyle.ASCII)
                .build();
        for (int word : allWords) {
            Collection<String> neighbors = word2Vec.wordsNearest(String.valueOf(word), maxKNN);
            Set<Integer> nearestWords = new HashSet<>(neighbors.stream().map(Integer::parseInt).collect(Collectors.toSet()));
            nearestWords = nearestWords.stream()
                    .filter(e -> word2Vec.similarity(String.valueOf(word), String.valueOf(e)) > minSim)
                    .collect(Collectors.toSet());
            nearestWords.add(word);
            if (nearestWords.isEmpty()) {
                continue;
            }
            String nearestString = nearestWords.stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(","));

            String freqQuery =
                    "SELECT id, word, tf " +
                            "FROM tf " +
                            "WHERE word IN (" + nearestString + ")";

            ResultSet freqRs = stmt.executeQuery(freqQuery);
            Map<String, Double> docToSum = new HashMap<>();
            while (freqRs.next()) {
                String docId = freqRs.getString("id");
                int sqlword = freqRs.getInt("word");
                double freq = freqRs.getDouble("tf");
                double similar = word2Vec.similarity(String.valueOf(word), String.valueOf(sqlword));
                double weighted = (word == sqlword ? freq : freq * lsim * similar);
                docToSum.merge(docId, weighted, Double::sum);
            }
            docToSum.forEach((did, tfsem) -> {
                try {

                    pstmtTFSem.setString(1, did);
                    pstmtTFSem.setInt(2, word);
                    pstmtTFSem.setDouble(3, tfsem);
                    pstmtTFSem.addBatch();

                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            });
            pstmtTFSem.executeBatch();
            pbis.step();
        }
        pbis.close();

        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tfsem_word ON tfsem(word)");
        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tfsem_id ON tfsem(id)");
        stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_tfsem_id_word ON tfsem(id, word)");

        String sqlIDFSem = "WITH document_count AS (\n" +
                "    SELECT COUNT(*) as total FROM documents\n" +
                "),\n" +
                "word_doc_counts AS (\n" +
                "    SELECT word, COUNT(DISTINCT id) as doc_count\n" +
                "    FROM tfsem\n" +
                "    GROUP BY word\n" +
                ")\n" +
                "UPDATE words\n" +
                "SET idfsem = LN(\n" +
                "    ((d.total - COALESCE(wdc.doc_count, 0) + 0.5) /\n" +
                "    (COALESCE(wdc.doc_count, 0) + 0.5)) + 1\n" +
                ")\n" +
                "FROM document_count d, word_doc_counts wdc\n" +
                "WHERE words.word = wdc.word;";
        stmt.executeUpdate(sqlIDFSem);
    }

    static void testBM25(Connection c) throws IOException, SQLException {
        List<QueryResult> queries = new ArrayList<>();
        double avgDoclen = 0;
        JSONArray jsDataBM25 = new JSONArray(Jsoup.parse(new File(testQuestionFile)).text());
        try (Statement statement = c.createStatement();
             ResultSet resultSet = statement.executeQuery("SELECT AVG(doclen) AS avg_doclen FROM documents")) {
            if (resultSet.next()) {
                avgDoclen = resultSet.getDouble("avg_doclen");
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        for (int i = 0; i < jsDataBM25.length(); i++) {
            JSONObject questBM25Obj = jsDataBM25.getJSONObject(i);
            String query = questBM25Obj.getString("query");
            Encoding encoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE);
            Map<String, Double> Ans = computeBM25Scores(encoding, query, c, BM25_k, BM25_b, avgDoclen, false);
            Map<String, Double> semAns = computeBM25Scores(encoding, query, c, BM25_k, BM25_b, avgDoclen, true);
            Map<String, Integer> relevantAns = new HashMap<>();

            ArrayList<String> resultBM25 = new ArrayList<>(Ans.entrySet().stream()
                    .sorted((entry1, entry2) -> Double.compare(entry2.getValue(), entry1.getValue()))
                    .limit(TOP_k)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList()));

            ArrayList<String> resultBM25Sem = new ArrayList<>(semAns.entrySet().stream()
                    .sorted((entry1, entry2) -> Double.compare(entry2.getValue(), entry1.getValue()))
                    .limit(TOP_k)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList()));

            JSONArray jsDataBM25Rdocs = questBM25Obj.getJSONArray("relevant_docs");
            for (int u = 0; u < jsDataBM25Rdocs.length(); u++) {
                JSONObject jsDataBM25RdocsObj = jsDataBM25Rdocs.getJSONObject(u);
                relevantAns.put(jsDataBM25RdocsObj.getString("doc_id"), jsDataBM25RdocsObj.getInt("score"));
            }

            queries.add(new QueryResult(query, resultBM25, resultBM25Sem, relevantAns));
        }

        compare(queries, TOP_k);
    }

    static Connection getConnection() throws SQLException {
        try {
            Class.forName("org.postgresql.Driver");
        } catch (ClassNotFoundException e) {
            System.err.println("PostgreSQL JDBC Driver not found.");
            e.printStackTrace();
        }

        Properties props = new Properties();
        props.setProperty("user", USER);
        props.setProperty("password", PASSWORD);

        return DriverManager.getConnection(URL_DATABASE, props);
    }

    public static Map<String, Double> computeBM25Scores(Encoding encoding, String query, Connection conn, double k, double b, double avg_doclen, boolean semOn) throws SQLException {
        Map<String, List<TFIDFValue>> docTfIdfMap = new HashMap<>();
        IntArrayList tokenIds = encoding.encode(query);

        Map<Integer, Double> idfMap = new HashMap<>();// Getting the idf for all the words in the query
        try (PreparedStatement ps = conn.prepareStatement(
                //"SELECT word, "+(semOn ? "idfsem":"idf")+" FROM words WHERE word = ANY (?)")) {
                "SELECT word, idf FROM words WHERE word = ANY (?)")) {
            Array sqlArray = conn.createArrayOf("integer", tokenIds.boxed().toArray());
            ps.setArray(1, sqlArray);
            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                //idfMap.put(rs.getInt("word"), rs.getDouble(semOn ? "idfsem":"idf"));
                idfMap.put(rs.getInt("word"), rs.getDouble("idf"));
            }
        }

        try (PreparedStatement ps = conn.prepareStatement( // We get tf (frequency) for documents (the query selects only those doc_ids that have records with these words)
                "SELECT id, word, " + (semOn ? "tfsem" : "tf") + " FROM " + (semOn ? "tfsem" : "tf") + " WHERE word = ANY (?)")) {
            Array sqlArray = conn.createArrayOf("integer", tokenIds.boxed().toArray());
            ps.setArray(1, sqlArray);
            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                String docId = rs.getString("id");
                int word = rs.getInt("word");
                double tf = rs.getDouble(semOn ? "tfsem" : "tf");
                double idf = idfMap.getOrDefault(word, 0.0);
                docTfIdfMap.computeIfAbsent(docId, d -> new ArrayList<>())
                        .add(new TFIDFValue(word, tf, idf));
            }
        }

        Map<String, Integer> docLengths = new HashMap<>();
        try (PreparedStatement ps = conn.prepareStatement(// We get the length of all the documents that met
                "SELECT id, doclen FROM documents WHERE id = ANY (?)")) {
            Array docArray = conn.createArrayOf("text", docTfIdfMap.keySet().toArray());
            ps.setArray(1, docArray);
            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                docLengths.put(rs.getString("id"), rs.getInt("doclen"));
            }
        }

        Map<String, Double> bm25Scores = new HashMap<>();// Counting the final BM25
        for (var entry : docTfIdfMap.entrySet()) {
            bm25Scores.put(entry.getKey(), getBM25WeightForDoc(entry.getValue(), k, b, avg_doclen, docLengths.getOrDefault(entry.getKey(), 1)));
        }

        return bm25Scores;
    }

    static double getBM25WeightForDoc(List<TFIDFValue> tfidfList, double k, double b, double avDocLen, double docLen) {
        double sum = 0;
        for (TFIDFValue tfidf : tfidfList) {
            sum += tfidf.idf * ((tfidf.tf * (k + 1)) / tfidf.tf + (k * (1 - b + (b * (docLen / avDocLen)))));
        }
        return sum;
    }

    static class TFIDFValue {
        int word;
        double tf;
        double idf;

        TFIDFValue(int word, double tf, double idf) {
            this.tf = tf;
            this.idf = idf;
            this.word = word;
        }
    }

    public static void compare(List<QueryResult> queries, int k) {
        if (queries == null || queries.isEmpty()) {
            System.out.println("No data");
            return;
        }

        double pA = meanPrecisionAtK(queries, true, k);
        double rA = meanRecallAtK(queries, true, k);
        double f1A = meanF1AtK(queries, true, k);
        double mapA = meanAveragePrecision(queries, true, k);
        double ndcgA = meanNdcgAtK(queries, true, k);
        double scoreA = sumScore(queries, true, k);
        double score2A = sumScore2(queries, true, k);

        double pB = meanPrecisionAtK(queries, false, k);
        double rB = meanRecallAtK(queries, false, k);
        double f1B = meanF1AtK(queries, false, k);
        double mapB = meanAveragePrecision(queries, false, k);
        double ndcgB = meanNdcgAtK(queries, false, k);
        double scoreB = sumScore(queries, false, k);
        double score2B = sumScore2(queries, false, k);

        System.out.printf("(top-%d)", k);
        System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s%n", "Алгоритм", "Precision", "Recall", "F1", "MAP", "nDCG");
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "A", pA, rA, f1A, mapA, ndcgA, scoreA, score2A);
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "B", pB, rB, f1B, mapB, ndcgB, scoreB, score2B);
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "Delta(A-B)", pA - pB, rA - rB, f1A - f1B, mapA - mapB, ndcgA - ndcgB, scoreA - scoreB, score2A - score2B);

        System.out.println("\n++++++++++++++++++++++++++++++++++++++++++");
        pA = 0;
        rA = 0;
        f1A = 0;
        mapA = 0;
        ndcgA = 0;
        scoreA = 0;
        score2A = 0;

        pB = 0;
        rB = 0;
        f1B = 0;
        mapB = 0;
        ndcgB = 0;
        scoreB = 0;
        score2B = 0;
        int tr = 0;
        for (QueryResult qr : queries) {
            double pqa = precisionAtK(qr.resultsA, qr.relevance, k);
            double pqb = precisionAtK(qr.resultsB, qr.relevance, k);
            double rqa = recallAtK(qr.resultsA, qr.relevance, k);
            double rqb = recallAtK(qr.resultsB, qr.relevance, k);
            double f1qa = f1AtK(qr.resultsA, qr.relevance, k);
            double f1qb = f1AtK(qr.resultsB, qr.relevance, k);
            double mapqa = averagePrecision(qr.resultsA, qr.relevance, k);
            double mapqb = averagePrecision(qr.resultsB, qr.relevance, k);
            double ndcga = ndcgAtK(qr.resultsA, qr.relevance, k);
            double ndcgb = ndcgAtK(qr.resultsB, qr.relevance, k);
            double scorea = score(qr.resultsA, qr.relevance, k);
            double scoreb = score(qr.resultsB, qr.relevance, k);
            double score2a = score2(qr.resultsA, qr.relevance, k);
            double score2b = score2(qr.resultsB, qr.relevance, k);

            if (scorea != scoreb) {
                pA += pqa;
                rA += rqa;
                f1A += f1qa;
                mapA += mapqa;
                ndcgA += ndcga;
                scoreA += scorea;
                score2A += score2a;

                pB += pqb;
                rB += rqb;
                f1B += f1qb;
                mapB += mapqb;
                ndcgB += ndcgb;
                scoreB += scoreb;
                score2B += score2b;
                tr++;
            }
        }
        pA = pA / tr;
        rA = rA / tr;
        f1A = f1A / tr;
        mapA = mapA / tr;
        ndcgA = ndcgA / tr;

        pB = pB / tr;
        rB = rB / tr;
        f1B = f1B / tr;
        mapB = mapB / tr;
        ndcgB = ndcgB / tr;

        System.out.println("Only the different ones");
        System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s%n", "Type", "Precision", "Recall", "F1", "MAP", "nDCG", "Score");
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "A", pA, rA, f1A, mapA, ndcgA, scoreA);
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "B", pB, rB, f1B, mapB, ndcgB, scoreB);
        System.out.printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n", "Delta(A-B)", pA - pB, rA - rB, f1A - f1B, mapA - mapB, ndcgA - ndcgB, scoreA - scoreB);
    }

    public static class QueryResult {
        public final String query;
        public final List<String> resultsA;
        public final List<String> resultsB;
        /**
         * Map docId -> relevance score (1 - good, 2 - very good )
         */
        public final Map<String, Integer> relevance;

        public QueryResult(String query, List<String> resultsA, List<String> resultsB, Map<String, Integer> relevance) {
            this.query = query;
            this.resultsA = resultsA == null ? Collections.emptyList() : resultsA;
            this.resultsB = resultsB == null ? Collections.emptyList() : resultsB;
            this.relevance = relevance == null ? Collections.emptyMap() : relevance;
        }
    }

    /**
     * Returns a sublist of the first k elements (or less if there are fewer results)
     */
    private static List<String> topK(List<String> list, int k) {
        if (list == null) return Collections.emptyList();
        if (k <= 0) return Collections.emptyList();
        return list.size() <= k ? list : list.subList(0, k);
    }

    public static double precisionAtK(List<String> retrieved, Map<String, Integer> relevance, int k) {
        List<String> top = topK(retrieved, k);
        if (top.isEmpty()) return 0.0;
        int hit = 0;
        for (String doc : top) if (relevance.containsKey(doc)) hit++;
        return (double) hit / top.size();
    }

    public static double recallAtK(List<String> retrieved, Map<String, Integer> relevance, int k) {
        List<String> top = topK(retrieved, k);
        int hit = 0;
        for (String doc : top) if (relevance.containsKey(doc)) hit++;
        long totalRelevant = relevance.size();
        return totalRelevant == 0 ? 0.0 : (double) hit / (double) totalRelevant;
    }

    public static double f1AtK(List<String> retrieved, Map<String, Integer> relevance, int k) {
        double p = precisionAtK(retrieved, relevance, k);
        double r = recallAtK(retrieved, relevance, k);
        return (p + r) == 0.0 ? 0.0 : 2 * p * r / (p + r);
    }

    /**
     * Average Precision (AP) is the classic version: we count binary (>= relThreshold).
     * We normalize by the number of relevant documents (>= threshold).
     */
    public static double averagePrecision(List<String> retrieved, Map<String, Integer> relevance, int k) {
        List<String> top = topK(retrieved, k);
        int hit = 0;
        double sumPrec = 0.0;
        for (int i = 0; i < top.size(); i++) {
            String doc = top.get(i);
            if (relevance.containsKey(doc)) {
                hit++;
                sumPrec += (double) hit / (i + 1);
            }
        }
        long totalRelevant = relevance.size();
        return totalRelevant == 0 ? 0.0 : sumPrec / (double) totalRelevant;
    }

    /**
     * DCG@K for graded relevance.
     * An exponential contribution formula is used: (2^rel - 1) / log2(i+2)
     */
    public static double dcgAtK(List<String> retrieved, Map<String, Integer> relevance, int k) {
        List<String> top = topK(retrieved, k);
        double dcg = 0.0;
        for (int i = 0; i < top.size(); i++) {
            String doc = top.get(i);
            int rel = relevance.getOrDefault(doc, 0);
            if (rel > 0) {
                double gain = Math.pow(2.0, rel) - 1.0;
                dcg += gain / (Math.log(i + 2) / Math.log(2)); // log2(i+2)
            }
        }
        return dcg;
    }

    /**
     * IDCG@K: The ideal DCG is to sort all relevant documents in descending order of relevance and take the top K.
     */
    public static double idcgAtK(Map<String, Integer> relevance, int k) {
        if (relevance == null || relevance.isEmpty()) return 0.0;
        List<Integer> scores = relevance.values().stream()
                .filter(v -> v > 0)
                .sorted(Comparator.reverseOrder())
                .collect(Collectors.toList());
        double idcg = 0.0;
        for (int i = 0; i < Math.min(scores.size(), k); i++) {
            int rel = scores.get(i);
            double gain = Math.pow(2.0, rel) - 1.0;
            idcg += gain / (Math.log(i + 2) / Math.log(2));
        }
        return idcg;
    }

    public static double ndcgAtK(List<String> retrieved, Map<String, Integer> relevance, int k) {
        double dcg = dcgAtK(retrieved, relevance, k);
        double idcg = idcgAtK(relevance, k);
        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    public static double score(List<String> retrieved, Map<String, Integer> relevance, int k) {
        if (retrieved.isEmpty()) return 0.0;
        int score = 0;
        for (String doc : retrieved) if (relevance.containsKey(doc)) score += relevance.get(doc);
        return (double) score;
    }

    public static double score2(List<String> retrieved, Map<String, Integer> relevance, int k) {
        if (retrieved.isEmpty()) return 0.0;
        int score = 0;
        for (String doc : retrieved) if (relevance.containsKey(doc) && relevance.get(doc) == 2) score++;
        return (double) score;
    }

    public static double meanPrecisionAtK(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += precisionAtK(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return queries.isEmpty() ? 0.0 : sum / queries.size();
    }

    public static double meanRecallAtK(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += recallAtK(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return queries.isEmpty() ? 0.0 : sum / queries.size();
    }

    public static double meanF1AtK(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += f1AtK(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return queries.isEmpty() ? 0.0 : sum / queries.size();
    }

    public static double meanAveragePrecision(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries)
            sum += averagePrecision(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return queries.isEmpty() ? 0.0 : sum / queries.size();
    }

    public static double meanNdcgAtK(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += ndcgAtK(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return queries.isEmpty() ? 0.0 : sum / queries.size();
    }

    public static double sumScore(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += score(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return sum;
    }

    public static double sumScore2(List<QueryResult> queries, boolean useFirstAlg, int k) {
        double sum = 0.0;
        for (QueryResult qr : queries) sum += score2(useFirstAlg ? qr.resultsA : qr.resultsB, qr.relevance, k);
        return sum;
    }
}
