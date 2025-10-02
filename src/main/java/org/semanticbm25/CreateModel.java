package org.semanticbm25;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This class creates a Word2vec model and saves it to a file.
 * dataPath - the path to the documents for training the model. (documents should be processed in DocumentTokenizer).
 * savePath - the path for saving the model.
 * It is possible to configure training parameters in the getFitModel method.(more information about the setup <a href="https://deeplearning4j.konduit.ai/deeplearning4j/reference/word2vec-glove-doc2vec#neural-word-embeddings">...</a>)
 *
 * @author Aleksei Shrank
 * @version 1.0
 * @since 2025-09-01
 */

public class CreateModel {

    public static String dataPath = "\\BEIR NFCorpus\\DocToken";
    public static String savePath = "\\BEIR NFCorpus\\model300Token.bin";

    public static void main(String[] args) {
        System.setProperty("org.nd4j.linalg.defaultbackend", "org.nd4j.linalg.cpu.nativecpu.CpuBackend");
        System.out.println("Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        System.setProperty("org.nd4j.linalg.memory.directbuffer", "true");

        getFitModel();
    }

    public static Word2Vec getFitModel() {
        SentenceIterator iter = getIterator(dataPath);
        TokenizerFactory tokenizer = getTokenizer();

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .layerSize(300)
                .seed(29)
                .windowSize(15)
                .iterate(iter)
                .tokenizerFactory(tokenizer)
                .workers(8)
                .build();

        vec.fit();
        WordVectorSerializer.writeWord2VecModel(vec, new File(savePath));
        return vec;
    }

    public static SentenceIterator getIterator(String directoryPath) {
        return new FileSentenceIterator(new File(directoryPath));
    }

    public static TokenizerFactory getTokenizer() {
        NumericTokenizerFactory tokenizer = new NumericTokenizerFactory();
        return tokenizer;
    }

    public static class NumericTokenizerFactory implements TokenizerFactory {
        @Override
        public Tokenizer create(String sentence) {
            return new NumericTokenizer(sentence);
        }

        @Override
        public Tokenizer create(InputStream inputStream) {
            try {
                String text = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
                        .lines()
                        .collect(Collectors.joining("\n"));
                return new NumericTokenizer(text);
            } catch (Exception e) {
                throw new RuntimeException("Error reading InputStream: " + e.getMessage(), e);
            }
        }

        @Override
        public TokenPreProcess getTokenPreProcessor() {
            return null;
        }

        @Override
        public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {

        }
    }

    public static class NumericTokenizer implements Tokenizer {
        private final List<String> tokens;
        private int currentIndex;

        public NumericTokenizer(String sentence) {
            this.tokens = Arrays.asList(sentence.trim().split("\\s+"));
            this.currentIndex = 0;
        }

        @Override
        public boolean hasMoreTokens() {
            return currentIndex < tokens.size();
        }

        @Override
        public int countTokens() {
            return 0;
        }

        @Override
        public String nextToken() {
            return tokens.get(currentIndex++);
        }

        @Override
        public List<String> getTokens() {
            return tokens;
        }

        @Override
        public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {

        }
    }
}
