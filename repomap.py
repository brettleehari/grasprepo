import colorsys
import math
import os
import random
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from utils import dump, filter_important_files, Spinner

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())

SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)

class RepoMap:
    def __init__(self, map_mul_no_files=8):
        self.map_mul_no_files = map_mul_no_files


    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        print(f"Generating repo map with chat_files: {chat_files}, other_files: {other_files}")
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        print("Calling get_ranked_tags_map...")
        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            print("Error: Disabling repo map, git repo too large?", file=sys.stderr)
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            print(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            print(f"Warning: File not found error: {fname}", file=sys.stderr)



    def get_tags(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
            print(f"Parsing file: {fname} with language: {lang}")

        except Exception as err:
            print(f"Warning: Skipping file {fname}: {err}", file=sys.stderr)
            return

        query_scm = get_scm_fname(lang)
        if not query_scm:
            return
        query_scm = query_scm.read_text()

        try:
            with open(fname, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading {fname}: {e}", file=sys.stderr)
            return
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(self, list_of_files):
        import networkx as nx
        from pyvis.network import Network


        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()



        list_of_files = sorted(list_of_files)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(list_of_files)
        #sys.exit(0) #breakpoint-1


        for fname in list_of_files:


            # dump(fname)
            rel_fname = self.get_rel_fname(fname)


            tags = list(self.get_tags(fname, rel_fname))
            print(f"Tags for file: {rel_fname} are: {tags}")
            #sys.exit(0) #breakpoint-2

            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        dump(defines)
        dump(references)
        dump(personalization)
        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:


            definers = defines[ident]
            if ident.startswith("_"):
                mul = 0.1
            #if ident in mentioned_idents: #if we want to give more weight to the mentioned idents
            #    mul = 10
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)
        # Initialize Pyvis network and convert NetworkX graph to it

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()
        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            # Issue #1536
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:


            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        dump(ranked_definitions)
        # Compute degree centrality (or any other ranking metric)
        centrality = nx.degree_centrality(G)

        # Find the node with the highest centrality
        highest_centrality_node = max(centrality, key=centrality.get)

        # Initialize a Pyvis Network object
        net = Network(notebook=True)

        # Add nodes with different colors based on their rank
        for node in G.nodes():
            if node == highest_centrality_node:
                # Mark the highest-ranked node in red
                net.add_node(node, label=f"Node {node}", color="red", size=25, title="Highest Rank")
            else:
                # Other nodes in blue
                net.add_node(node, label=f"Node {node}", color="blue", size=15)

        # Add edges from the NetworkX graph
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        graphname = "graph"+str(time.time())+".html"
        net.show(graphname)
        for node, neighbors in G.adjacency():
            print(f"Node {node}: {list(neighbors)}")
        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            #if fname in chat_rel_fnames:
            #    continue
            ranked_tags += list(definitions.get((fname, ident), []))
        print("##################################")
        print(ranked_tags)

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        print(top_rank)
        #sys.exit(0) #breakpoint-3
        for rank, fname in top_rank:
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))
        print("############ranked tags######################")
        print(ranked_tags)
        #sys.exit(0) #breakpoint-4
        print("############ranked tags above######################")
        return ranked_tags



    def render_tree(self, abs_fname, rel_fname, lois):
        print(f"Rendering tree for file: {rel_fname} with lines of interest: {lois}")

        # Read the file content directly
        try:
            with open(abs_fname, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading {abs_fname}: {e}", file=sys.stderr)
            return ""

        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
        )

        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        return res
    
    def get_tree(self, tags):
        ranked_tags_fnames = set(tag[0] for tag in tags)
        self.tree_cache = dict()

    def to_tree(self, tags):

        print(f"Converting tags to tree structure, number of tags: {len(tags)}")


        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            #if this_rel_fname in chat_rel_fnames:
            #    continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    queries_dir = Path(__file__).parent / "queries"
    print(f"Getting tags query file for language: {lang}")
    print(f"Queries dir: {queries_dir}")    
    scm_file = queries_dir / f"tree-sitter-{lang}-tags.scm"
    if scm_file.exists():
        return scm_file
    return None


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res

def parse_dir(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    print(f"Found {len(src_files)} source files in {directory}")
    return src_files



def main():

    unpacked = parse_dir(sys.argv[1])


    rm = RepoMap()
    rank_tags = rm.get_ranked_tags(unpacked)
    #import pdb; pdb.set_trace()
    repo_tree = rm.to_tree(rank_tags)
    print(repo_tree)

if __name__ == "__main__":
    main()


