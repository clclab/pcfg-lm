package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import base.Edge;
import base.FragmentRule;
import base.ProbRule;
import base.RuleSet;

import cern.colt.matrix.DoubleMatrix2D;

import parser.Completion;
import parser.EdgeSpace;
import parser.Prediction;

import edu.stanford.nlp.parser.lexparser.IntTaggedWord;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Distribution;
import edu.stanford.nlp.stats.TwoDimensionalCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Index;

public class Util {
  public static DecimalFormat df = new DecimalFormat("0.0");
  public static DecimalFormat df1 = new DecimalFormat("0.0000");
  public static DecimalFormat df3 = new DecimalFormat("000");
  
  public static void error(boolean cond, String message){
  	if(cond){
  		System.err.println(message);
  		System.exit(1);
  	}
  }
  
  /**
   * Returns the intersection of sets s1 and s2 (use set sizes to determine intersection order
   */
  public static <E> Set<E> intersection(Set<E> s1, Set<E> s2) {
    Set<E> s = Generics.newHashSet();
    if(s1.size()<s2.size()){
      for(E e : s1){
        if (s2.contains(e)){
          s.add(e);
        }
      }
    } else {
      for(E e : s2){
        if (s1.contains(e)){
          s.add(e);
        }
      }
    }
    
    return s;
  }
  
  public static boolean isEqual(int[] values1, int[] values2){
    if (values1.length != values2.length){
      return false;
    } else {
      for (int i = 0; i < values1.length; i++) {
        if(values1[i] != values2[i]){
          return false;
        }
      }
      return true;
    }
  }
  
  public static List<Integer> toList(int[] values){
    List<Integer> valueList = new ArrayList<Integer>(values.length);
    for (int value : values) {
      valueList.add(value);
    }
    
    return valueList;
  }
  
  public static int[] subArray(int[] values, int start, int end){
    if (end<=start){
      return new int[0];
    }
    
    int[] newValues = new int[end-start];
    if (end - start >= 0) System.arraycopy(values, start, newValues, start - start, end - start);
    return newValues;
  }
  
  public static BufferedReader getBufferedReaderFromFile(String inFile){
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile)));
    } catch (FileNotFoundException e) {
      System.err.println("! File not found: " + inFile);
    } 
    return br;
  }
  
  public static BufferedReader getBufferedReaderFromString(String str) throws FileNotFoundException{
    return new BufferedReader(new StringReader(str));
  }

  public static int[][] permutationMatrix(int size){
    // C(n, i): # ways of choosing i numbers from n numbers
    // C(n, 0) = C(n, n) = 1
    // C(n, i) = C(n-1, i) + C(n-1, i-1)
    
    int[][] c = new int[size+1][size+1];
    
    // init
    for (int n = 0; n <=size; n++) {
      c[n][0] = 1;
      c[n][n] = 1;
    }
    
    for (int n = 2; n<=size; n++) {
      for (int i = 1; i < n; i++) {
        c[n][i] = c[n-1][i] + c[n-1][i-1];
      }
    }
    
//    for (int i = 0; i < c.length; i++) {
//      for (int j = 0; j <=i; j++) {
//        System.err.print(df3.format(c[i][j]) + " ");
//      }
//      System.err.println();
//    }
    return c;
  }
  public static List<Integer> getNonterminals(Map<Integer, Integer> nonterminalMap){
    List<Map.Entry<Integer, Integer>> list = new LinkedList<Map.Entry<Integer, Integer>>(nonterminalMap.entrySet());

    Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
      public int compare(Map.Entry<Integer, Integer> m1, Map.Entry<Integer, Integer> m2) {
        return (m1.getValue()).compareTo(m2.getValue());
      }
    });

    List<Integer> result = new LinkedList<Integer>();
    for (Map.Entry<Integer, Integer> entry : list) {
        result.add(entry.getKey());
    }
    
    return result;
  }
  /**
   * returns a collection of scored rules corresponding to all non-terminal productions from a collection of trees.
   */
  public static Collection<ProbRule> tagRulesFromTrees(Collection<Tree> trees, 
      Index<String> motherIndex, Index<String> childIndex, Map<Integer, Integer> nonterminalMap) {
    Collection<ProbRule> rules = new ArrayList<ProbRule>();
    TwoDimensionalCounter<Integer, List<Integer>> ruleCounts = 
      new TwoDimensionalCounter<Integer, List<Integer>>();
    
    // go through trees
    for(Tree tree:trees){
      for(Tree subTree : tree.subTreeList()){
        if (subTree.isLeaf() || subTree.isPreTerminal()) { // ignore leaf and preterminal nodes
          continue;
        }
     
        // increase count
        int index = motherIndex.indexOf(subTree.value(), true);
        ruleCounts.incrementCount(index, 
            getChildrenFromTree(subTree.children(), childIndex));
        
        // add nonterminals
        if(!nonterminalMap.containsKey(index)){
          nonterminalMap.put(index, nonterminalMap.size());
        }
      }
    }

    for(int mother: ruleCounts.firstKeySet()){ // go through all rules
      // normalize w.r.t to parent node
      Distribution<List<Integer>> normalizedChildren = 
        Distribution.getDistribution(ruleCounts.getCounter(mother));
      for(List<Integer> childList : normalizedChildren.keySet()){
        rules.add(new ProbRule(new FragmentRule(mother, childList, true), normalizedChildren.getCount(childList)));
      }
    }

    return rules;
  }
  

  private static List<Integer> getChildrenFromTree(Tree[] trees, Index<String> childIndex) {
    List<Integer> children = new ArrayList<Integer>(trees.length);
    for (int i = 0; i < trees.length; i++) {
      Tree tree = trees[i];
      children.add(childIndex.indexOf(tree.value(), true));
    }
    return children;
  }
  
  /**
   * Load a file into a list of strings, one line per string
   * 
   * @param inFile
   * @return
   * @throws IOException
   */
  public static List<String> loadFile(String inFile) throws IOException{
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile)));
    List<String> lines = new ArrayList<String>();
    
    String line = null;
    while((line = br.readLine()) != null)
      lines.add(line);
    
    br.close();
    return lines;
  }
  
  /**
   * Output results for each sentence, each word corresponds to a number
   * 
   * @param sentenceString
   * @param outWriter
   * @param results
   * @throws IOException
   */
  public static void outputSentenceResult(String sentenceString, 
      BufferedWriter outWriter, List<Double> results) throws IOException {
    List<String> sentence = Arrays.asList(sentenceString.split("\\s+"));
    for (int i = 0; i < sentence.size(); i++) {
      outWriter.write(sentence.get(i) + " " + results.get(i) + "\n");
    }
    outWriter.write("#! Done\n");
    outWriter.flush();
  }
  
  public static void init(double[] dl, double value) {
    for (int i = 0; i < dl.length; i++) {
      dl[i] = value;
    }
  }
  
  public static void init(double[][] dl, double value) {
    for (int j = 0; j < dl.length; j++) {
      init(dl[j], value);
    }
  }
  
  public static void init(double[][][] dl, double value) {
    for (int left = 0; left < dl.length; left++) {
//      init(dl[left], value);
      
      for (int right = left; right < dl[left].length; right++) {
        init(dl[left][right], value);
      }
    }
  }
  
  /** Print to string methods **/
  // print double array
  public static String sprint(double[] values){
    StringBuffer sb = new StringBuffer("[");
    
    if(values.length > 0){
      for(double value : values){
        sb.append(value + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append("]");
    return sb.toString();
  }


  // print double list
  public static String sprint(List<Double> values){
  	StringBuffer sb = new StringBuffer("[");

  	if(values.size() > 0){
  		for(double value : values){
  			sb.append(value + ", ");
  		}
  		sb.delete(sb.length()-2, sb.length());
  	}
  	sb.append("]");
  	return sb.toString();
  }

  public static String sprintProb(double[] values, Operator operator){
    StringBuffer sb = new StringBuffer("[");
    
    if(values.length > 0){
      for(double value : values){
        sb.append(operator.getProb(value) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append("]");
    return sb.toString();
  }
  
  // print boolean array
  public static String sprint(boolean[] values){
    StringBuffer sb = new StringBuffer("[");

    for (boolean value : values) {
      sb.append(value + ", ");
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }

  public static String sprint(Index<String> index){
    StringBuffer sb = new StringBuffer("[");
    
    if(index.size() > 0){
      for (int i = 0; i < index.size(); i++) {
        sb.append(index.get(i) + ", ");
      }
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(Index<Edge> edgeIndex, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("[");
    
    if(edgeIndex.size() > 0){
      for (int i = 0; i < edgeIndex.size(); i++) {
        sb.append(edgeIndex.get(i).toString(tagIndex, wordIndex) + ", ");
      }
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(Set<Integer> edges, EdgeSpace edgeSpace, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("[");
    
    if(edges.size() > 0){
      for (Integer edge : edges) {
        sb.append(edgeSpace.get(edge).toString(tagIndex, wordIndex) + ", ");
      }
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(List<Integer> edges, EdgeSpace edgeSpace, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("[");
    
    if(edges.size() > 0){
      for (Integer edge : edges) {
        sb.append(edgeSpace.get(edge).toString(tagIndex, wordIndex) + ", ");
      }
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
//  public static String sprintEdgeMap(Map<Integer, Set<Integer>> edgeMap, EdgeSpace edgeSpace, Index<String> tagIndex, Index<String> wordIndex){
//    StringBuffer sb = new StringBuffer("[");
//    
//    List<Integer> sortedIndices = new ArrayList<Integer>(edgeMap.keySet());
//    Collections.sort(sortedIndices);
//    if(sortedIndices.size() > 0){
//      for (Integer edge : sortedIndices) {
//        sb.append(Util.sprint(edgeMap.get(edge), edgeSpace, tagIndex, wordIndex) + "\n");
//      }
//    }
//    return sb.toString();
//  }
  
  public static String sprint(Map<Integer, Double> valueMap, Index<String> tagIndex){
    StringBuffer sb = new StringBuffer("(");
    
    if(valueMap.size() > 0){
      for (int iT : valueMap.keySet()) {
        double score = valueMap.get(iT);
        if (score<=0){
          score = Math.exp(score);
        }
        sb.append(tagIndex.get(iT) + "=" + df1.format(score) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append(")");
    return sb.toString();
  }
  
  public static String sprint(Map<String, Double> valueMap){
    StringBuffer sb = new StringBuffer("(");
    
    if(valueMap.size() > 0){
      for (String key : valueMap.keySet()) {
        double score = valueMap.get(key);
        
        sb.append(key + "=" + df1.format(score) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append(")");
    return sb.toString();
  }
  
  public static String sprint(Map<Integer, Double> valueMap, RuleSet ruleSet
      , Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("(");
    
    if(valueMap.size() > 0){
      for (int ruleId : valueMap.keySet()) {
        double score = valueMap.get(ruleId);
        if (score<=0){
          score = Math.exp(score);
        }
        sb.append(ruleSet.get(ruleId).toString(tagIndex, wordIndex) + "=" + df1.format(score) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append(")");
    return sb.toString();
  }
  
  // print Prediction[]
  public static String sprint(Prediction[] predictions, EdgeSpace edgeSpace, 
      Index<String> tagIndex, Index<String> wordIndex, Operator operator){
    StringBuffer sb = new StringBuffer("(");
    for(Prediction prediction : predictions){
      sb.append(prediction.toString(edgeSpace, tagIndex, wordIndex, operator) + ", ");
    }
    if (predictions.length > 0) {
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append(")");
    return sb.toString();
  }

  // print Completion[]
  public static String sprint(Completion[] completions, EdgeSpace edgeSpace, 
      Index<String> tagIndex, Index<String> wordIndex, Operator operator){
    StringBuffer sb = new StringBuffer("[");
    for(Completion completion : completions){
      sb.append(completion.toString(edgeSpace, tagIndex, wordIndex, operator) + ", ");
    }
    if (completions.length > 0) {
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(Index<String> tagIndex, Collection<Integer> indices){
    StringBuffer sb = new StringBuffer("[");
    for(int index : indices){
      //sb.append("(" + index + ", " + tagIndex.get(index) + ") ");
      sb.append(tagIndex.get(index) + ", ");
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(Index<String> tagIndex, int[] indices){
    StringBuffer sb = new StringBuffer("[");
    for(int index : indices){
      //sb.append("(" + index + ", " + tagIndex.get(index) + ") ");
      sb.append(tagIndex.get(index) + ", ");
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  
  public static String sprint(Map<Integer, Map<Integer, Double>> unaryEdgeMap, 
      EdgeSpace edgeSpace, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("{");
    for(int tag : unaryEdgeMap.keySet()){
      //sb.append("(" + index + ", " + tagIndex.get(index) + ") ");
      sb.append(tagIndex.get(tag) + "={");
      for(int edge : unaryEdgeMap.get(tag).keySet()){
        sb.append(edgeSpace.get(edge).toString(tagIndex, wordIndex) + 
            "=" + unaryEdgeMap.get(tag).get(edge) + ", ");
      }
      if(unaryEdgeMap.get(tag).keySet().size()>0){
        sb.delete(sb.length()-2, sb.length());
      }
      sb.append("}, ");
    }
    
    if(unaryEdgeMap.keySet().size()>0){
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append("}");
    return sb.toString();
  }
  
  public static String sprint(Collection<ProbRule> rules, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer();
    for(ProbRule rule : rules){
      sb.append(rule.toString(tagIndex, wordIndex) + "\n");
    }
    return sb.toString();
  }
  
  public static String schemeSprint(Collection<ProbRule> rules, Index<String> tagIndex, Index<String> wordIndex){
    StringBuffer sb = new StringBuffer("[");
    for(ProbRule rule : rules){
      sb.append(rule.schemeString(tagIndex, wordIndex) + ", ");
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(Map<Integer, Counter<Integer>> int2intCounter
      , Index<String> keyIndex, Index<String> valueIndex){
    StringBuffer sb = new StringBuffer("{");
    for(int iKey : int2intCounter.keySet()){
      Counter<Integer> counter = int2intCounter.get(iKey);
      sb.append(keyIndex.get(iKey) + "={");
      for(int iValue : counter.keySet()){
        sb.append(valueIndex.get(iValue) + "=" + counter.getCount(iValue) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
      sb.append("}, ");
    }
    if(int2intCounter.size()>0) {
    	sb.delete(sb.length()-2, sb.length());
    }
    return sb.toString();
  }

  public static String sprint(Set<Integer> indices, Index<String> index){
    StringBuffer sb = new StringBuffer("[");
    
    if(indices.size() > 0){
      for (Integer edge : indices) {
        sb.append(index.get(edge) + ", ");
      }
    }
    sb.delete(sb.length()-2, sb.length());
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprintWord2Tags(Map<Integer, Set<IntTaggedWord>> word2tagsMap
      , Index<String> wordIndex, Index<String> tagIndex){
    StringBuffer sb = new StringBuffer("{");
    for(int iW : word2tagsMap.keySet()){
      Set<IntTaggedWord> itwSet = word2tagsMap.get(iW);
      sb.append(wordIndex.get(iW) + "=[");
      for(IntTaggedWord itw : itwSet){
        sb.append(itw.toString(wordIndex, tagIndex) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
      sb.append("}, ");
    }
    sb.delete(sb.length()-2, sb.length());
    return sb.toString();
  }
  
  public static String sprint(Set<IntTaggedWord> itws, 
      Index<String> wordIndex, Index<String> tagIndex){
    StringBuffer sb = new StringBuffer("[");
    if(itws.size()>0){
      for(IntTaggedWord itw : itws){
        sb.append(itw.toString(wordIndex, tagIndex) + ", ");
      }
      sb.delete(sb.length()-2, sb.length());
    }
    sb.append("]");
    return sb.toString();
  }
  
  public static String sprint(DoubleMatrix2D matrix){
    StringBuffer sb = new StringBuffer();
    int numRows = matrix.rows();
    int numCols = matrix.columns();
    
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        sb.append(matrix.get(i, j) + " ");
      }
      sb.deleteCharAt(sb.length()-1);
      sb.append("\n");
    }
    sb.deleteCharAt(sb.length()-1);
    
    return sb.toString();
  }  
}
