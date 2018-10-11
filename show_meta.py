from os.path import splitext
import sys

from asciitree import BoxStyle, LeftAligned
import h5py
from s3fs import S3FileSystem
from s3fs.mapping import S3Map
import zarr
from zarr.util import TreeNode, TreeTraversal, TreeViewer


def tree(file):
    return ChunkedTreeViewer(file)


class ChunkedTreeNode(TreeNode):
    def __init__(self, obj, depth=0, level=None):
        TreeNode.__init__(self, obj, depth, level)

    def get_children(self):
        if hasattr(self.obj, "values"):
            if self.level is None or self.depth < self.level:
                depth = self.depth + 1
                return [
                    ChunkedTreeNode(o, depth=depth, level=self.level)
                    for o in self.obj.values()
                ]
        return []

    def get_text(self):
        name = self.obj.name.split("/")[-1] or "/"
        if hasattr(self.obj, "shape"):
            name += " {} {} chunks={}".format(
                self.obj.shape, self.obj.dtype, self.obj.chunks
            )
        return name


class ChunkedTreeViewer(TreeViewer):
    def __init__(self, group, expand=False, level=None):
        TreeViewer.__init__(self, group, expand, level)

    def __unicode__(self):
        drawer = LeftAligned(
            traverse=TreeTraversal(),
            draw=BoxStyle(gfx=self.unicode_kwargs, **self.text_kwargs),
        )
        root = ChunkedTreeNode(self.group, level=self.level)
        return drawer(root)


def make_store(path):
    if path.startswith("s3://"):
        s3 = S3FileSystem()
        return S3Map(path[len("s3://") :], s3=s3)

    return zarr.DirectoryStore(path)


def show_meta(input):
    input_path, input_ext = splitext(input)
    if input_ext == ".h5" or input_ext == ".h5ad" or input_ext == ".loom":
        file = h5py.File(input, "r")
        return zarr.tree(file)
    elif input_ext == ".zarr":
        store = make_store(input)
        file = zarr.open(store)
        return tree(file)


if __name__ == "__main__":
    input = sys.argv[1].rstrip("/")
    meta = show_meta(input)
    print(meta)
