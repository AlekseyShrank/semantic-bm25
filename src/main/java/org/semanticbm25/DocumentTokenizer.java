package org.semanticbm25;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class processes txt documents. He changes the text to tokens for learning Word2vec.
 *
 * @author Aleksei Shrank
 * @version 1.0
 * @since 2025-09-01
 */

public class DocumentTokenizer {
    public static void main(String[] args) {
        String inputDir = "\\BEIR NFCorpus\\Doc";
        String outputDir = "\\BEIR NFCorpus\\DocToken";

        Encoding encoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE);
        System.out.println("BPE tokenizer has been initialized (cl100k_base).");

        try {
            Files.createDirectories(Paths.get(outputDir));
        } catch (IOException e) {
            System.err.println("Create dir error: " + e.getMessage());
            return;
        }

        try {
            ProgressBar pb = new ProgressBarBuilder()
                    .setTaskName("Doc tokenized")
                    .setInitialMax(Files.list(Paths.get(inputDir)).count())
                    .setStyle(ProgressBarStyle.ASCII)
                    .build();
            Files.list(Paths.get(inputDir))
                    .filter(path -> path.toString().endsWith(".txt"))
                    .forEach(path -> processFile(path, outputDir, encoding, pb));
            pb.close();
        } catch (IOException e) {
            System.err.println("Read dir error: " + e.getMessage());
        }
    }

    private static void processFile(Path inputPath, String outputDir, Encoding encoding, ProgressBar pb) {
        try {
            String text = Files.readString(inputPath);
            List<String> sentences = splitIntoSentences(text);
            List<String> tokenizedSentences = new ArrayList<>();
            for (String sentence : sentences) {
                if (!sentence.trim().isEmpty()) {
                    IntArrayList tokenIds = encoding.encode(sentence);
                    String tokenizedSentence = IntStream.range(0, tokenIds.size())
                            .mapToObj(tokenIds::get)
                            .map(String::valueOf)
                            .collect(Collectors.joining(" "));
                    tokenizedSentences.add(tokenizedSentence);
                }
            }

            String tokenizedText = String.join("\n", tokenizedSentences);
            Path outputPath = Paths.get(outputDir, inputPath.getFileName().toString());
            Files.writeString(outputPath, tokenizedText);
            pb.step();

        } catch (IOException e) {
            System.err.println("Processed file error " + inputPath.getFileName() + ": " + e.getMessage());
        }
    }

    private static List<String> splitIntoSentences(String text) {
        List<String> sentences = new ArrayList<>();
        BreakIterator iterator = BreakIterator.getSentenceInstance(Locale.US);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            String sentence = text.substring(start, end).trim();
            if (!sentence.isEmpty()) {
                sentences.add(sentence);
            }
        }
        return sentences;
    }
}
