from __future__ import annotations
from typing import List, TypedDict, Tuple, Optional
from typing_extensions import NotRequired
import argparse
import pathlib
import bvh
import glm
import json
from enum import Enum, auto


class HumanBones(Enum):
    # 6
    Hips = auto()
    Spine = auto()
    Chest = auto()
    UpperChest = auto()
    Neck = auto()
    Head = auto()
    # 8
    LeftUpperLeg = auto()
    LeftLowerLeg = auto()
    LeftFoot = auto()
    LeftToes = auto()
    RightUpperLeg = auto()
    RightLowerLeg = auto()
    RightFoot = auto()
    RightToes = auto()
    # 8
    LeftShoulder = auto()
    LeftUpperArm = auto()
    LeftLowerArm = auto()
    LeftHand = auto()
    RightShoulder = auto()
    RightUpperArm = auto()
    RightLowerArm = auto()
    RightHand = auto()


def guess_humanoid(name: str) -> Optional[HumanBones]:
    match (name.lower()):
        case "hips":
            return HumanBones.Hips
        case "spine":
            return HumanBones.Spine
        case "spine1":
            return HumanBones.Chest
        case "neck":
            return HumanBones.Neck
        case "head":
            return HumanBones.Head
        case "leftshoulder":
            return HumanBones.LeftShoulder
        case "leftarm":
            return HumanBones.LeftUpperArm
        case "leftforearm":
            return HumanBones.LeftLowerArm
        case "lefthand":
            return HumanBones.LeftHand
        case "rightshoulder":
            return HumanBones.RightShoulder
        case "rightarm":
            return HumanBones.RightUpperArm
        case "rightforearm":
            return HumanBones.RightLowerArm
        case "righthand":
            return HumanBones.RightHand
        case "leftupleg":
            return HumanBones.LeftUpperLeg
        case "leftleg":
            return HumanBones.LeftLowerLeg
        case "leftfoot":
            return HumanBones.LeftFoot
        case "lefttoebase":
            return HumanBones.LeftToes
        case "rightupleg":
            return HumanBones.RightUpperLeg
        case "rightleg":
            return HumanBones.RightLowerLeg
        case "rightfoot":
            return HumanBones.RightFoot
        case "righttoebase":
            return HumanBones.RightToes
    print(f"## {name} not found ##")


class GltfAsset(TypedDict):
    version: str


class GltfScene(TypedDict):
    nodes: List[int]


class GltfNode(TypedDict):
    name: str
    translation: Tuple[float, float, float]
    children: NotRequired[List[int]]


class Gltf(TypedDict):
    asset: GltfAsset
    scene: int
    scenes: List[GltfScene]
    nodes: List[GltfNode]


def print_bvh(node, indent=""):
    print(f"{indent}{node}")
    for child in node.children:
        print_bvh(child, indent + "  ")


def print_gltf(nodes, node_index, indent=""):
    node = nodes[node_index]
    print(f"{indent}{node['name']}: {node['translation']}")
    for child_index in node.get("children", []):
        print_gltf(nodes, child_index, indent + "  ")


def build_hierarchy(
    gltf_nodes: List[GltfNode], gltf_parent: GltfNode, bvh_node: bvh.BvhNode
):
    offset = bvh_node["OFFSET"]
    gltf_node: GltfNode = {
        "name": bvh_node.name,
        "translation": (float(offset[0]), float(offset[1]), float(offset[2])),
    }
    gltf_child_index = len(gltf_nodes)
    if "children" not in gltf_parent:
        gltf_parent["children"] = []
    gltf_nodes.append(gltf_node)
    gltf_parent["children"].append(gltf_child_index)

    for child in bvh_node.filter("JOINT"):
        build_hierarchy(gltf_nodes, gltf_node, child)


class GetBB:
    def __init__(self, gltf: Gltf) -> None:
        self.gltf = gltf
        self.min = glm.vec3(float("inf"), float("inf"), float("inf"))
        self.max = glm.vec3(-float("inf"), -float("inf"), -float("inf"))

    def traverse(self, parent=glm.vec3(0, 0, 0), node_index=0, indent=""):
        node = self.gltf["nodes"][node_index]
        offset = node["translation"]
        pos = parent + glm.vec3(offset[0], offset[1], offset[2])
        self.min = glm.vec3(
            min(self.min.x, pos.x), min(self.min.y, pos.y), min(self.min.z, pos.z)
        )
        self.max = glm.vec3(
            max(self.max.x, pos.x), max(self.max.y, pos.y), max(self.max.z, pos.z)
        )
        # print(f'{indent}{node["name"]}: {pos}')
        for child_index in node.get("children", []):
            self.traverse(pos, child_index, indent + "  ")


def convert(src: pathlib.Path, dst: pathlib.Path):
    root: GltfNode = {
        "name": "__ROOT__",
        "translation": (0, 0, 0),
    }
    nodes: List[GltfNode] = [root]
    gltf: Gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
    }

    mocap = bvh.Bvh(src.read_text())
    bvh_root = next(mocap.root.filter("ROOT"))
    build_hierarchy(nodes, root, bvh_root)
    # print_bvh(bvh_root)
    print_gltf(gltf["nodes"], 0)
    bb = GetBB(gltf)
    bb.traverse()
    print(bb.min, bb.max)
    # get_bb(gltf)
    dst.write_text(json.dumps(gltf), encoding="utf-8")

    map = {}
    for node in gltf["nodes"][1:]:
        bone = guess_humanoid(node["name"])
        if bone:
            print(f'{node["name"]} => {bone}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh", help="src bvh path", type=pathlib.Path)
    parser.add_argument("vrm", help="dst vrm path", type=pathlib.Path)
    args = parser.parse_args()
    convert(args.bvh, args.vrm)


if __name__ == "__main__":
    main()
