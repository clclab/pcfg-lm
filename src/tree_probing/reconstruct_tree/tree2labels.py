
from nltk.tree import Tree
import copy
import itertools
from numpy import insert
from collections import Counter
from numpy.core.defchararray import index
import logging

logger = logging.getLogger(__name__)

"""
Class to manage the transformation of a constituent tree into a sequence of labels
and vice versa. It extends the Tree class from the NLTK framework to address constituent Parsing as a 
sequential labeling problem.
"""
class SeqTree(Tree):
    
    EMPTY_LABEL = "EMPTY-LABEL"
    
    def __init__(self,label,children):
         
        self.encoding = None
        super(SeqTree, self).__init__(label,children) 
    
    # At the moment only the RelativeLevelTreeEncoder is supported
    def set_encoding(self, encoding):
        self.encoding = encoding

    """
    Transforms a constituent tree with N leaves into a sequence of N labels.
    @param is_binary: True if binary trees are being encoded and want to use an optimized
    encoding [Not tested at the moment]
    @param root_label: Set to true to include a special label to words directly attached to the root
    @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
    """
    def to_maxincommon_sequence(self,is_binary=False, root_label=False, encode_unary_leaf=False,
                                abs_top=None, abs_neg_gap=None):
        
        if self.encoding is None: raise ValueError("encoding attribute is None")
        leaves_paths = []
        self.path_to_leaves([self.label()],leaves_paths)
        leaves = self.leaves()
        unary_sequence =  [s.label() for s in self.subtrees(lambda t: t.height() == 2)] #.split("+")
        return self.encoding.to_maxincommon_sequence(leaves, leaves_paths, unary_sequence, binarized=is_binary, 
                                                     root_label= root_label,
                                                     encode_unary_leaf=encode_unary_leaf,
                                                     abs_top=abs_top,
                                                     abs_neg_gap=abs_neg_gap)


    @classmethod
    def maxincommon_to_tree(cls, sequence, sentence, encoding):
        """
        Transforms a predicted sequence into a constituent tree
        @params sequence: A list of the predictions 
        @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
        @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
        concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
        """
        if encoding is None: raise ValueError("encoding parameter is None")
        # At the moment only the RelativeLevelTreeEncoder is supported (see RelativeLevelTreeEncoder class)
        return encoding.maxincommon_to_tree(sequence, sentence)


    """
    Gets the path from the root to each leaf node
    Returns: A list of lists with the sequence of non-terminals to reach each 
    terminal node
    """
    def path_to_leaves(self, current_path, paths):

        for i, child in enumerate(self):
            
            pathi = []
            if isinstance(child,Tree):
                common_path = copy.deepcopy(current_path)
                
                #common_path.append(child.label())
                common_path.append(child.label()+"*"+str(i))
                #common_path.append(child.label()+"-"+str(i))
                child.path_to_leaves(common_path, paths)
            else:
                for element in current_path:
                    pathi.append(element)
                pathi.append(child)
                paths.append(pathi)
    
        return paths
    
    
class RelativeLevelTreeEncoder(object):
    """
    Encoder/Decoder class to transform a constituent tree into a sequence of labels by representing
    how many levels in the tree there are in common between the word_i and word_(i+1) (in a relative scale) 
    and the label (constituent) at that lowest ancestor.
    """
    ROOT_LABEL = "ROOT"
    NONE_LABEL = "NONE"
    SPLIT_LABEL_SURNAME_SYMBOL = "*"


    def __init__(self, join_char="~",split_char="@"):
        self.join_char = join_char
        self.split_char = split_char


    def to_maxincommon_sequence(self, leaves, leaves_paths, unary_sequence, 
                                binarized, root_label, encode_unary_leaf=False,
                                abs_top=None, abs_neg_gap=None):
        """
        Transforms a tree into a sequence encoding the "deepest-in-common" phrase between words t and t+1
        @param leaves: A list of words representing each leaf node
        @param leaves_paths: A list of lists that encodes the path in the tree to reach each leaf node
        @param unary_sequence: A list of the unary sequences (if any) for every leaf node
        @param binarized: If True, when predicting an "ascending" level we map the tag to -1, as it is possible to determine in which
        level the word t needs to be located
        @param root_label: Set to true to include a special label ROOT to the words that are directly attached to the root of the sentence
        @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
        """  
        sequence = []
        previous_ni = 0
        ni=0
        relative_ni = 0 
        previous_relative_ni=0
        
        for j,leaf in enumerate(leaves):
            
            #It is the last real word of the sentence
            if j == len(leaves)-1: 
                
                #NEWJOINT     
                if encode_unary_leaf and self.join_char in unary_sequence[j]:
                    encoded_unary_leaf = self.split_char+self.join_char.join(unary_sequence[j].split(self.join_char)[:-1]) #The PoS tags is not encoded
                else:
                    encoded_unary_leaf = ""          
#                 if encode_unary_leaf and "+" in unary_sequence[j]:
#                     encoded_unary_leaf = "_"+"+".join(unary_sequence[j].split("+")[:-1]) #The PoS tags is not encoded
#                 else:
#                     encoded_unary_leaf = ""
 
 
 
#               #This corresponds to the implementation without the computation trick
#                sequence.append((self.NONE_LABEL+encoded_unary_leaf))
#                break

                #TODO: This is a computation trick that seemed to work better in the dev set
                #Sentences of length on are annotated with ROOT_UNARYCHAIN instead NONE_UNARYCHAIN                 
                if (root_label and len(leaves)==1):
                    #TODO: Check if this is working for the combined "top-down" encoding
                    #sequence.append(self.ROOT_LABEL+encoded_unary_leaf)
                    sequence.append("1"+self.ROOT_LABEL+encoded_unary_leaf)
                else:
                    sequence.append((self.NONE_LABEL+encoded_unary_leaf))
                break
            
            explore_up_to = min( len(leaves_paths[j]), len(leaves_paths[j+1]) )+1   
            ni = 0

            for i in range(explore_up_to):   
                     
                if leaves_paths[j][i] == leaves_paths[j+1][i]:
                    ni+=1
                else:              
                    relative_ni = ni - previous_ni              
                    if binarized:
                        relative_ni = relative_ni if relative_ni >=0 else -1
                        
                    #NEWJOINT
                    if encode_unary_leaf and self.join_char in unary_sequence[j]:
                        encoded_unary_leaf = self.split_char+self.join_char.join(unary_sequence[j].split(self.join_char)[:-1]) #The PoS tags is not encoded
                    else:
                        encoded_unary_leaf = ""

#                     if encode_unary_leaf and "+" in unary_sequence[j]:
#                         encoded_unary_leaf = "_"+"+".join(unary_sequence[j].split("+")[:-1]) #The PoS tags is not encoded
#                     else:
#                         encoded_unary_leaf = ""

                    
                    #The root_label is activated and it is a top two level
                   # print root_label, ni, abs_top, relative_ni, abs_neg_gap
                    if (root_label and ni==1) or (abs_top is not None and 
                                                  abs_neg_gap is not None and 
                                                  root_label and ni < (abs_top+1) 
                                                  and relative_ni <= abs_neg_gap): #and ni==1:

                        #NEWJOINT
                        sequence.append(str(ni)+self.ROOT_LABEL+self.split_char+leaves_paths[j][ni-1].split(self.SPLIT_LABEL_SURNAME_SYMBOL)[0]+encoded_unary_leaf)
                   #     sequence.append(str(ni)+self.ROOT_LABEL+"_"+leaves_paths[j][ni-1].split(self.SPLIT_LABEL_SURNAME_SYMBOL)[0]+encoded_unary_leaf)
                        
                    else:
                        sequence.append(self._tag(relative_ni, leaves_paths[j][ni-1])+encoded_unary_leaf)
                    
                    previous_ni = ni
                    previous_relative_ni = relative_ni
                    break

     #   exit()
        return sequence   

    
    def uncollapse(self, tree):
        """
        Uncollapses the INTERMEDIATE unary chains and also removes empty nodes that might be created when
        transforming a predicted sequence into a tree.
        @precondition: Uncollapsing/Removing-empty from the root must be have done prior to to call 
        this function
        """
        uncollapsed = []
        for child in tree:

            if type(child) == type(u'') or type(child) == type(""):
                uncollapsed.append(child)
            else:
                #It also removes EMPTY nodes
                while child.label() == SeqTree.EMPTY_LABEL and len(child) != 0:
                    child = child[-1]
                
                label = child.label()
                #NEWJOINT
                if self.join_char in label:
                #if '+' in label: #and label[-1] != "+": #To support SPMRL datasets
                     
                    #NEWJOINT
                    label_split = label.split(self.join_char) 
                    #label_split = label.split('+')
                    swap = Tree(label_split[0],[])

                    last_swap_level = swap
                    for unary in label_split[1:]:
                        last_swap_level.append(Tree(unary,[]))
                        last_swap_level = last_swap_level[-1]
                    last_swap_level.extend(child)
                    uncollapsed.append(self.uncollapse(swap))
                #We are uncollapsing the child node
                else:     
                    uncollapsed.append(self.uncollapse(child))
        
        tree = Tree(tree.label(),uncollapsed)
        return tree
    
    
    def get_postag_trees(self,tree):
        """
        Gets a list of the PoS tags from the tree
        @return A list containing the PoS tags
        """
        postags = []
        
        for nchild, child in enumerate(tree):
            
            if len(child) == 1 and type(child[-1]) == type(""):
                postags.append(child)
            else:
                postags.extend(self.get_postag_trees(child))
        
        return postags


    def preprocess_tags(self,pred):
        """
        Transforms a prediction of the form LEVEL_LABEL_[UNARY_CHAIN] into a tuple
        of the form (level,label):
        level is an integer or None (if the label is NONE or NONE_leafunarychain).
        label is the constituent at that level
        @return (level, label)
        """
        try:         
            # NEWJOINT
            label = pred.split(self.split_char)
            level, label = label[0],label[1]  
            try:
                return (int(level), label)
            except ValueError:
                
                #It is a NONE label with a leaf unary chain
                if level == self.NONE_LABEL:
                    return (None,pred.rsplit(self.split_char,1)[1])
                
                return (level,label)
            
        except IndexError:
            # It is a NONE label (without any leaf unary chains)
            return (None, pred)
        
        
    def maxincommon_to_tree(self, sequence, sentence):
        """
        Transforms a predicted sequence into a constituent tree
        @params sequence: A list of the predictions 
        @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
        @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
        concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
        """
        tree = SeqTree(SeqTree.EMPTY_LABEL,[])
        current_level = tree
        previous_at = None
        first = True

        sequence = list(map(self.preprocess_tags, sequence))
        sequence = self._to_absolute_encoding(sequence)      

        for j, (level,label) in enumerate(sequence):

            if level is None:
                prev_level, _ = sequence[j-1]
                previous_at = tree
                while prev_level is not None and prev_level > 1:
                    previous_at = previous_at[-1]
                    prev_level-=1
          
                #It is a NONE label
                if self.NONE_LABEL == label:

                    previous_at.append( Tree( sentence[j][1],[ sentence[j][0]]) )
                #It is a leaf unary chain
                #NEWJOINT
                else:
                    
                    if label[0].isdigit() and self.ROOT_LABEL in label:

                        previous_at.append(Tree(self.join_char+sentence[j][1],[ sentence[j][0]]))

                    else:
                        previous_at.append(Tree(label+self.join_char+sentence[j][1],[ sentence[j][0]]))   

                return tree
                   
            i=0
            for i in range(level-1):
                if len(current_level) == 0 or i >= sequence[j-1][0]-1: 
                    child_tree = Tree(SeqTree.EMPTY_LABEL,[])                      
                    current_level.append(child_tree)   
                    current_level = child_tree

                else:
                    current_level = current_level[-1]
                    
            if current_level.label() == SeqTree.EMPTY_LABEL:    
                current_level.set_label(label)
                        
            if first:
                previous_at = current_level
                previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))
                first=False
            else:         
                #If we are at the same or deeper level than in the previous step
                if i >= sequence[j-1][0]-1: 
                    current_level.append(Tree( sentence[j][1],[sentence[j][0]]))
                else:
                    previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))        
                previous_at = current_level
                
            current_level = tree
            
        return tree


    def _to_absolute_encoding(self, relative_sequence):
        """
        Transforms an encoding of a tree in a relative scale into an
        encoding of the tree in an absolute scale.
        :param relative_sequence: A list of tuples (level,label) representing the encoding of a tree in a relative scale
        """
        absolute_sequence = [0] * len(relative_sequence)
        current_level = 0
        for j, (level,phrase) in enumerate(relative_sequence):
        
            if level is None:
                # None is added to even the length of the sequence and the sentence
                absolute_sequence[j] = (level, phrase)

            elif type(level) == type("") and self.ROOT_LABEL in level:
                # Set ROOT to 1 and and current level to 1
                try:
                    aux_level = int(level.replace(self.ROOT_LABEL,""))
                    absolute_sequence[j] = (aux_level, phrase)
                except ValueError:
                    aux_level = 1
                    absolute_sequence[j] = (aux_level,phrase)

                current_level = aux_level

            else:
                # Keep counting                
                current_level+= level
                absolute_sequence[j] = (current_level,phrase)

        return absolute_sequence
    
    def _tag(self,level,tag):
        #NEWJOINT
        return str(level)+self.split_char+tag.rsplit("*",1)[0]