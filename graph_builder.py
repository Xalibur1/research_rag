import os
import networkx as nx
from schemas import PaperExtraction

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_extraction(self, extraction: PaperExtraction):
        paper_id = extraction.paper_id or "Unknown"
        paper_node_id = f"Paper:{paper_id}"
        self.graph.add_node(paper_node_id, type="Paper", title=extraction.title or "Unknown", authors=extraction.authors, year=extraction.year or "Unknown")

        if not extraction.method:
            return self.graph

        method_name = extraction.method.name if extraction.method.name and extraction.method.name.strip() else "ProposedMethod"
        method_node_id = f"Method:{method_name}"
        self.graph.add_node(method_node_id, type="Method", purpose=extraction.method.purpose_one_sentence or "")
        self.graph.add_edge(paper_node_id, method_node_id, relation="proposes")

        for inp in extraction.method.inputs:
            input_node = f"InputType:{inp.name or 'Unknown'}"
            self.graph.add_node(input_node, type="InputType", dtype=inp.type or "", shape=inp.shape_or_example or "")
            self.graph.add_edge(method_node_id, input_node, relation="takes_input")

        for out in extraction.method.outputs:
            out_node = f"OutputType:{out.name or 'Unknown'}"
            self.graph.add_node(out_node, type="OutputType", dtype=out.type or "", shape=out.shape_or_example or "")
            self.graph.add_edge(method_node_id, out_node, relation="produces")

        if extraction.evaluation:
            for ds in extraction.evaluation.datasets:
                ds_node = f"Dataset:{ds}"
                self.graph.add_node(ds_node, type="Dataset")
                self.graph.add_edge(method_node_id, ds_node, relation="evaluated_on")

            for metric in extraction.evaluation.metrics:
                m_node = f"Metric:{metric}"
                self.graph.add_node(m_node, type="Metric")
                self.graph.add_edge(method_node_id, m_node, relation="reports_metric")

            for res in extraction.evaluation.results:
                ds_val = res.dataset or "Unknown"
                met_val = res.metric or "Unknown"
                val_val = res.value or "Unknown"
                res_node = f"Result:{ds_val}_{met_val}_{val_val}"
                self.graph.add_node(res_node, type="Result", value=val_val, dataset=ds_val, metric=met_val)
                self.graph.add_edge(method_node_id, res_node, relation="reports")
                if res.baseline_name:
                    base_node = f"Baseline:{res.baseline_name}"
                    self.graph.add_node(base_node, type="Baseline")
                    self.graph.add_edge(res_node, base_node, relation="compared_to")

        for lim in extraction.method.limitations:
            lim_node = f"Limitation:{lim[:30]}..."
            self.graph.add_node(lim_node, type="Limitation", full_text=lim)
            self.graph.add_edge(paper_node_id, lim_node, relation="has")

        for assump in extraction.method.assumptions:
            assump_node = f"Assumption:{assump[:30]}..."
            self.graph.add_node(assump_node, type="Assumption", full_text=assump)
            self.graph.add_edge(paper_node_id, assump_node, relation="has")

        # Code/Data links
        for link in extraction.method.code_or_data_links:
            link_node = f"Resource:{link}"
            self.graph.add_node(link_node, type="Resource")
            self.graph.add_edge(paper_node_id, link_node, relation="provides")

        return self.graph
    
    def get_neighborhood(self, node_id: str, depth: int = 1) -> nx.DiGraph:
        if node_id not in self.graph:
            return nx.DiGraph()
        return nx.ego_graph(self.graph, node_id, radius=depth, undirected=True)

    def describe_graph(self) -> str:
        description = f"Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.\n"
        for u, v, data in self.graph.edges(data=True):
            description += f"{u} --[{data.get('relation', '-')}]--> {v}\n"
        return description
