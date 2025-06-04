from nltk.corpus import BracketParseCorpusReader
import re


np_regex = re.compile('[^N]*NP.*')
vp_regex = re.compile('VP.*')
pp_regex = re.compile('PP.*')
tr_leaf_regex = re.compile('\*\-[0-9]*|0|\*')
ix_label_regex = re.compile('.*\-[0-9]+')

def load_ptb(file_pattern, corpus_root=r"resources/PennTreebank/"):
    ptb = BracketParseCorpusReader(corpus_root, file_pattern)
    return ptb

######################## 
# NLTK tree methods
######################## 

def preprocess_only_parentheses(sent_notr):
    preproc_alignment = dict()
    sent_preproc = []
    for m, tok in enumerate(sent_notr):
        preproc_alignment[m] = m
        tok = tok.replace('LRB', '(')
        tok = tok.replace('LCB', '(')
        tok = tok.replace('RRB', ')')
        tok = tok.replace('RCB', ')')
        if ('(' in tok or ')' in tok) and len(tok) > 1:
            print('not fully covered: ', tok)
        sent_preproc.append(tok)

    return preproc_alignment, sent_preproc



def preprocess(sent_notr, only_parentheses=False):
    """
    Return a tuple of preproc_alignment, preproc_sent
    where preproc_alignment is a dict from preproc_sent to sent_notr
    and preproc_sent is a list of strings that represents the preprocessed sentence (using space as sep. for list elements)

    If only_parentheses=True, the alignment is one-to-one, the tokenisation stays the same and only RRB/LRB labels are replaced
    """

    if only_parentheses:
        return preprocess_only_parentheses(sent_notr)

    preproc_alignment = dict()
    preproc_sent = []
    m = 0 # current token
    last_main_token_preproc = 0
    last_main_token_notr = 0
    current_preproc_token = ''
    skip_because_suffix = 0
    while m < len(sent_notr):
        current_notr_token = sent_notr[m]

        # skip:
        """
        if current_notr_token == 'LRB' or current_notr_token == 'RRB':
            if m > 0:
                preproc_alignment[last_main_token_preproc] = last_main_token_notr-1
                last_main_token_preproc += 1
                last_main_token_notr += skip_because_suffix + 1
                skip_because_suffix = 0
                # preproc_sent.append(current_preproc_token)
        """    
        
        # attach to preceding: 
        # punctuation
        # closing quotes
        # possessive s
        # n't
        attach_prev_list = ["'re", "'s", ',', '.', 'RCB', 'RRB', "n't"]
        attach_next_list = ['LCB', 'LCB``', 'LRB', 'LRB``', '``']
        if current_notr_token in attach_prev_list or (current_notr_token == "''" and m>0):
            current_notr_token.replace('RRB',')')
            current_preproc_token += current_notr_token
            last_main_token_notr -= 1
            # last_main_token_preproc += 1
            skip_because_suffix += 1
    
        # attach to next:
        # opening quotes
        elif current_preproc_token in attach_next_list:             
            # preproc_sent.append(current_preproc_token)
            current_preproc_token += current_notr_token
        else: 
            if m > 0:
                preproc_alignment[last_main_token_preproc] = last_main_token_notr-1
                last_main_token_preproc += 1
                last_main_token_notr += skip_because_suffix
                skip_because_suffix = 0
                current_preproc_token = current_preproc_token.replace('LRB', '(')
                current_preproc_token = current_preproc_token.replace('LCB', '(')
                current_preproc_token = current_preproc_token.replace('RRB', ')')
                current_preproc_token = current_preproc_token.replace('RCB', ')')
                preproc_sent.append(current_preproc_token)
            current_preproc_token = current_notr_token
        last_main_token_notr += 1
        m = m + 1
        
    # append last token
    preproc_alignment[last_main_token_preproc] = last_main_token_notr-1
    preproc_sent.append(current_preproc_token)

    return preproc_alignment, preproc_sent


def find_trace_ix(i, tree_tr):
    """find labels with a coindex somewhere above leaf i
    """

    return find_xps_above_i(i, tree_tr, ix_label_regex)



def find_tracing_alignment(tree_notr, tree_tr):
    """return an alignment dict leaves of tree without traces -> leaves of tree with traces"""
    alignment = dict()
    tr_index = 0
    leaves_notr = tree_notr.leaves()
    leaves_tr = tree_tr.leaves()
    for m, leaf in enumerate(leaves_notr):
        assert len(leaves_tr) > tr_index
        tr_leaf = leaves_tr[tr_index]
        if leaf == tr_leaf or leaf == tr_leaf.replace('-', ''):
            alignment[m] = tr_index
        elif tr_leaf_regex.match(leaves_tr[tr_index]):
            tr_index = tr_index + 1
            tr_leaf = leaves_tr[tr_index]
            if leaf == tr_leaf or leaf == tr_leaf.replace('-', ''):
                alignment[m] = tr_index
            else:
                tr_index = tr_index + 1
                tr_leaf = leaves_tr[tr_index]
                if leaf == tr_leaf or leaf == tr_leaf.replace('-', ''):
                    alignment[m] = tr_index
                else:
                    tr_index = tr_index + 1
                    if leaf == leaves_tr[tr_index]:
                        alignment[m] = tr_index
                    else:
                        print('case not implemented. Sent with traces: ', leaves_tr)
        else:
            print('case not implemented.')
        tr_index = tr_index + 1    
    return alignment


def address_is_xp(address, tree, xp_regex):
    subtree = tree
    for i in address:
        subtree = subtree[i]
    label = subtree.label()
    return xp_regex.match(label) is not None

def find_end_indices(i, tree, xps_started_at_i):
    """Given 
    - a start index i
    - a tree
    - and the adresses of xps above terminal node i that 'start at' i

    compute a dictionary {xp_adress -> end_index} 
    """
    result = dict()
    for xp in xps_started_at_i:
        # find the specific XP
        subtree = tree
        for j in xp:
            subtree = subtree[j]
        end = i + len(subtree.leaves()) - 1
        result[tuple(xp)] = end
    return result

def find_xps_above_i(i, tree, xp_regex):
    """given a word index i in a tree, return all tuples of tree_positions above leaf i whose labels matches the regex"""
    i_treepos = tree.leaf_treeposition(i)
    subtree = tree
    xps = []
    traversed = []
    for j in i_treepos[:-1]:
        label = subtree.label()
        if xp_regex.match(label):
            xps.append(traversed)
        traversed = traversed + [j]
        subtree = subtree[j]
    return xps

def lowest_phrase_above_leaf_i(i, tree, return_target_ga=False):
    ga_of_target = tree.treeposition_spanning_leaves(i,i+1)[:-2]
    node = find_node(tree, ga_of_target)
    label = node.label()
    # print(label)
    label = re.sub('[^A-Za-z]+', '', label)

    if not (label.endswith('P') or label in ['S', 'ATSBAR', 'ATSQ', 'ATINTJ', 'ATS', 'SQ', 'SBAR', 'SBARQ', 'SINV', 
                                             'FRAG', 'NAC', 'NX', 'INTJ', 'LST' , 'X', 'RRC', 'ATSINV', 'ATFRAG', 'PPLOC']):
        if label in ['PRT','PRN', 'ATPRT', 'ATPRN']:
            ga_of_target = ga_of_target[:-1]
            node = find_node(tree, ga_of_target)
            label = node.label()
        else:
            assert False, f'Unknown label with which the method wil fail: \
                \n label: {label} \n sentence: {" ".join(tree.leaves())} \n word: {tree.leaves()[i]} \n tree: {tree}'

    if return_target_ga:
        return label, node, ga_of_target
    return label, node

def find_node(tree, address):
    """return subtree at address"""

    subtree = tree
    for i in address:
        subtree = subtree[i]
    return subtree

def find_label(tree, address):
    """return label at address in tree"""

    subtree = tree
    for i in address:
        subtree = subtree[i]
    return subtree.label()

def common_xp_ancestor(i, j, tree, xp_regex):
    shared_spine = tree.treeposition_spanning_leaves(i,j+1)
    subtree = tree
    for k in shared_spine:
        if xp_regex.match(subtree.label()) is not None:
            return True
        subtree = subtree[k]
    return xp_regex.match(subtree.label()) is not None

def asymmetric_lowest_common_ancestor(i, j, tree, xp_regex):
    """Test if for two tokens i and j, their lowest common ancestor is X and X is the lowest XP dominating i
    
    """
    # check the lowest common ancestor
    if i < j:
        shared_spine = tree.treeposition_spanning_leaves(i,j+1)
    elif i > j:
        shared_spine = tree.treeposition_spanning_leaves(j,i+1)
    subtree = tree
    for k in shared_spine:
        subtree = subtree[k]
    if xp_regex.match(subtree.label()) is None:
        return False
    
    # check if the LCA is also the lowest XP dominating i
    i_treepos = tree.leaf_treeposition(i)
    for l in i_treepos[len(shared_spine):-1]:
        subtree = subtree[l]
        if xp_regex.match(subtree.label()) is not None:
            return False
    return True