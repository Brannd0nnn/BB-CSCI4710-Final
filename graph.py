import networkx as nx
import matplotlib.pyplot as plt
import textwrap

#Visual NetworkX graph of the spam classification results
def show_probability_graph(user_message, spam_probability, not_spam_probability):
    graph = nx.Graph()

    # Wrap long messages so they fit better
    wrapped_message = "\n".join(textwrap.wrap(user_message, width=20))
    #Create labels for spam and not spam nodes
    spam_node = f"Spam\n{spam_probability:.2f}%"
    not_spam_node = f"Not Spam\n{not_spam_probability:.2f}%"
    #Add the user's message as the center node
    graph.add_node(wrapped_message)
    #Add probability result nodes
    graph.add_node(spam_node)
    graph.add_node(not_spam_node)
    #Connect the user message to both probability outcomes
    graph.add_edge(wrapped_message, spam_node)
    graph.add_edge(wrapped_message, not_spam_node)

    #Position nodes for a cleaner display
    pos = {
        wrapped_message: (0, 0),
        spam_node: (-1.5, -1),
        not_spam_node: (1.5, -1)
    }
    #Set graph display size
    plt.figure(figsize=(10, 6))

    #Draw graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=5000,
        font_size=9
    )
    #Display graph
    plt.show()