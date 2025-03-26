import matplotlib.pyplot as plt
import networkx as nx
import json

def plot_rerank_tree(tree_json):
    G = nx.DiGraph()
    with open(tree_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for parent, children in data.items():
        for child in children:
            G.add_edge(parent, child)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title('Reranking Paths Tree', fontsize=14)
    plt.savefig('rerank_tree.png')
    plt.show()

# example
tree = {
    "context": ["candidate1", "candidate2", "candidate3"],
    "candidate1": ["candidate1_1", "candidate1_2"],
    "candidate2": ["candidate2_1"],
    "candidate3": []
}

with open('tree.json', 'w', encoding='utf-8') as f:
    json.dump(tree, f, ensure_ascii=False)

plot_rerank_tree('tree.json')
