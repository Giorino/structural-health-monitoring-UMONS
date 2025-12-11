#!/usr/bin/env python3
import os
import html

def generate_dot():
    """Generates a Graphviz DOT file for the Neural Network pipeline."""
    dot_content = """digraph G {
    rankdir=LR;
    splines=ortho;
    nodesep=0.6;
    ranksep=0.8;
    fontname="Arial";
    
    // Default styles
    node [shape=rect, style="filled,rounded", fontname="Arial", fontsize=10, penwidth=1];
    edge [fontname="Arial", fontsize=9];

    // --- Input Data ---
    node [fillcolor="#9FA8DA", shape=cylinder, label=<<B><FONT POINT-SIZE="12">RAW DATA</FONT></B><BR/>(Interrogator Files)>] raw_data;

    // --- Data Handling ---
    subgraph cluster_data {
        label="Data Processing Pipeline";
        style=dashed;
        color="#B0BEC5";
        
        node [fillcolor="#FFF9C4", shape=note]; // Yellowish
        
        prep [label=<<B><FONT POINT-SIZE="12">Signal Processing</FONT></B><BR/><BR/>- Load Raw Interrogator Data<BR/>- Median Filter (Window Size=7)<BR/>- Peak Detection (Prominence=0.1)>];
        merge [label=<<B><FONT POINT-SIZE="12">Data Merging</FONT></B><BR/><BR/>- Align Timestamps (Mid-point)<BR/>- Calculate Wavelength Shifts<BR/>- Merge with Metadata (Excel)>];
        split [label=<<B><FONT POINT-SIZE="12">Data Preparation</FONT></B><BR/><BR/>- Sliding Window (Seq Len=50)<BR/>- Split (70/15/15)<BR/>- StandardScaler (Fit on Train)>];
    }

    // --- Branch 1: Neural Network (CNN) ---
    subgraph cluster_dl {
        label="Neural Network Pipeline";
        style="rounded";
        bgcolor="#E3F2FD"; // Light Blue background

        node [fillcolor="#BBDEFB", shape=rect];
        dl_loader [label=<<B><FONT POINT-SIZE="12">DataLoaders</FONT></B><BR/>(Batch=16, Seq=50, Feat=9)>];
        
        node [fillcolor="#90CAF9", shape=rect];
        cnn_arch [label=<<B><FONT POINT-SIZE="12">CNN Architecture</FONT></B><BR/><BR/>- Input Transpose (B, 9, 50)<BR/>- Conv1d(32, k=3) + ReLU<BR/>- Conv1d(64, k=3) + ReLU<BR/>- Conv1d(128, k=3) + ReLU<BR/>- Global Avg Pool + Dropout(0.2)<BR/>- Linear(128 -> 4 Classes)>];

        node [fillcolor="#64B5F6", shape=rect];
        dl_train [label=<<B><FONT POINT-SIZE="12">Training Process</FONT></B><BR/><BR/>- Optimizer: Adam<BR/>- Loss: CrossEntropy<BR/>- Scheduler: ReduceLROnPlateau>];
        
        node [fillcolor="#42A5F5", shape=diamond];
        dl_eval [label=<<B><FONT POINT-SIZE="12">Validation Loss</FONT></B><BR/>Decreasing?>];
        
        node [fillcolor="#90CAF9", shape=diamond];
        check_patience [label=<<B><FONT POINT-SIZE="12">Patience &lt; 10?</FONT></B>>];
        
        node [fillcolor="#1E88E5", shape=rect, fontcolor=white];
        dl_stop [label=<<B><FONT POINT-SIZE="12">Early Stopping</FONT></B><BR/>(Stop Training)>];
        
        node [fillcolor="#1E88E5", shape=rect, fontcolor=white];
        dl_save [label=<<B><FONT POINT-SIZE="12">Save Best Model</FONT></B><BR/>(.pth)>];
    }

    // --- Edges ---
    raw_data -> prep;
    prep -> merge;
    merge -> split;

    // Split to DL
    split -> dl_loader [label="Scaled Sequences"];
    dl_loader -> cnn_arch;
    cnn_arch -> dl_train;
    dl_train -> dl_eval;
    dl_eval -> dl_save [label="Yes (New Best)"];
    dl_eval -> check_patience [label="No"];
    check_patience -> dl_train [label="Yes (Continue)"];
    check_patience -> dl_stop [label="No (Stop)"];
    dl_save -> dl_train [label="Next Epoch"];
}
"""
    with open("flowchart.dot", "w") as f:
        f.write(dot_content)
    print("Generated flowchart.dot")

def generate_mermaid():
    """Generates a Mermaid diagram for the Neural Network pipeline."""
    mermaid_content = """graph LR
    %% Styles
    classDef raw fill:#9FA8DA,stroke:#333,stroke-width:1px;
    classDef data fill:#FFF9C4,stroke:#333,stroke-width:1px;
    classDef dl fill:#BBDEFB,stroke:#333,stroke-width:1px;
    classDef dl_dark fill:#64B5F6,stroke:#333,stroke-width:1px;
    classDef ml fill:#E1BEE7,stroke:#333,stroke-width:1px;
    classDef ml_dark fill:#BA68C8,stroke:#333,stroke-width:1px,color:white;

    %% Input
    RAW("<b><font size=4>RAW DATA</font></b><br/>(Interrogator Files)"):::raw

    %% Data Handling
    subgraph Data_Process [Data Processing Pipeline]
        direction TB
        PREP("<b><font size=4>Signal Processing</font></b><br/>- Load Raw Interrogator Data<br/>- Median Filter (Window Size=7)<br/>- Peak Detection (Prominence=0.1)"):::data
        MERGE("<b><font size=4>Data Merging</font></b><br/>- Align Timestamps (Mid-point)<br/>- Calculate Wavelength Shifts<br/>- Merge with Metadata (Excel)"):::data
        SPLIT("<b><font size=4>Data Preparation</font></b><br/>- Sliding Window (Seq=50)<br/>- Split (70/15/15)<br/>- StandardScaler (Fit on Train)"):::data
    end

    %% Neural Network Branch
    subgraph DL_Pipeline [Neural Network Pipeline ]
        direction TB
        DL_LOAD("<b><font size=4>DataLoaders</font></b><br/>(Batch=16, Seq=50, Feat=9)"):::dl
        CNN_ARCH("<b><font size=4>CNN Architecture</font></b><br/>- Input Transpose (B, 9, 50)<br/>- Conv1d(32, k=3) + ReLU<br/>- Conv1d(64, k=3) + ReLU<br/>- Conv1d(128, k=3) + ReLU<br/>- Global Avg Pool + Dropout(0.2)<br/>- Linear(128 -> 4 Classes)"):::dl
        DL_TRAIN("<b><font size=4>Training Process</font></b><br/>- Optimizer: Adam<br/>- Loss: CrossEntropy<br/>- Scheduler: ReduceLROnPlateau"):::dl_dark
        DL_EVAL{"<b><font size=4>Validation Loss</font></b><br/>Decreasing?"}:::dl_dark
        CHECK_PAT{"<b><font size=4>Patience < 10?</font></b>"}:::dl_dark
        DL_STOP("<b><font size=4>Early Stopping</font></b><br/>(Stop Training)"):::dl_dark
        DL_SAVE("<b><font size=4>Save Best Model</font></b><br/>(.pth)"):::dl_dark
    end


    %% Connections
    RAW --> PREP
    PREP --> MERGE
    MERGE --> SPLIT

    %% Split to DL
    SPLIT -->|"Scaled Sequences"| DL_LOAD
    DL_LOAD --> CNN_ARCH
    CNN_ARCH --> DL_TRAIN
    DL_TRAIN --> DL_EVAL
    DL_EVAL -->|"Yes (New Best)"| DL_SAVE
    DL_EVAL -->|"No"| CHECK_PAT
    CHECK_PAT -->|"Yes (Continue)"| DL_TRAIN
    CHECK_PAT -->|"No (Stop)"| DL_STOP
    DL_SAVE -->|"Next Epoch"| DL_TRAIN
"""
    with open("flowchart.mmd", "w") as f:
        f.write(mermaid_content)
    print("Generated flowchart.mmd")

def generate_drawio():
    """Generates a Draw.io XML file with manual layout."""
    
    # Helper to create mxCell
    def create_node(id, value, x, y, width, height, style):
        # Escape value for XML
        value = html.escape(value).replace("&#x27;", "'")
        return f"""<mxCell id="{id}" value="{value}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
        </mxCell>"""

    def create_edge(id, source, target, label=""):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
        return f"""<mxCell id="{id}" value="{label}" style="{style}" edge="1" parent="1" source="{source}" target="{target}">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>"""

    # Styles
    style_raw = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;fillColor=#9FA8DA;strokeColor=#000000;"
    style_data = "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;fillColor=#FFF9C4;strokeColor=#000000;align=left;spacingLeft=10;"
    style_dl = "rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#000000;"
    style_dl_dark = "rounded=1;whiteSpace=wrap;html=1;fillColor=#64B5F6;strokeColor=#000000;"
    style_diamond = "rhombus;whiteSpace=wrap;html=1;fillColor=#42A5F5;strokeColor=#000000;"
    style_stop = "rounded=1;whiteSpace=wrap;html=1;fillColor=#1E88E5;fontColor=#ffffff;strokeColor=#000000;"

    # Content
    nodes = []
    edges = []
    
    # --- Row 1: Data Processing (Y=100) ---
    nodes.append(create_node("raw", "<b><font style='font-size: 14px'>RAW DATA</font></b><br>(Interrogator Files)", 40, 100, 120, 80, style_raw))
    
    # Container for Data Processing
    nodes.append("""<mxCell id="group_data" value="Data Processing Pipeline" style="group;dashed=1;strokeColor=#B0BEC5;" vertex="1" connectable="0" parent="1">
          <mxGeometry x="200" y="60" width="750" height="160" as="geometry" />
        </mxCell>""")
    
    nodes.append(create_node("prep", "<b><font style='font-size: 14px'>Signal Processing</font></b><br><br>- Load Raw Interrogator Data<br>- Median Filter (Window Size=7)<br>- Peak Detection (Prominence=0.1)", 220, 80, 220, 120, style_data))
    nodes.append(create_node("merge", "<b><font style='font-size: 14px'>Data Merging</font></b><br><br>- Align Timestamps (Mid-point)<br>- Calculate Wavelength Shifts<br>- Merge with Metadata (Excel)", 480, 80, 220, 120, style_data))
    nodes.append(create_node("split", "<b><font style='font-size: 14px'>Data Preparation</font></b><br><br>- Sliding Window (Seq Len=50)<br>- Split (70/15/15)<br>- StandardScaler (Fit on Train)", 740, 80, 200, 120, style_data))

    # --- Row 2: Neural Network (Y=400) ---
    # Container for DL
    nodes.append("""<mxCell id="group_dl" value="Neural Network Pipeline" style="group;fillColor=#E3F2FD;strokeColor=#000000;" vertex="1" connectable="0" parent="1">
          <mxGeometry x="40" y="300" width="1100" height="250" as="geometry" />
        </mxCell>""")

    nodes.append(create_node("dl_loader", "<b><font style='font-size: 14px'>DataLoaders</font></b><br>(Batch=16, Seq=50, Feat=9)", 60, 380, 160, 60, style_dl))
    nodes.append(create_node("cnn_arch", "<b><font style='font-size: 14px'>CNN Architecture</font></b><br><br>- Input Transpose (B, 9, 50)<br>- Conv1d(32, k=3) + ReLU<br>- Conv1d(64, k=3) + ReLU<br>- Conv1d(128, k=3) + ReLU<br>- Global Avg Pool + Dropout(0.2)<br>- Linear(128 -> 4 Classes)", 260, 330, 240, 160, "rounded=1;whiteSpace=wrap;html=1;fillColor=#90CAF9;strokeColor=#000000;align=left;spacingLeft=10;"))
    nodes.append(create_node("dl_train", "<b><font style='font-size: 14px'>Training Process</font></b><br><br>- Optimizer: Adam<br>- Loss: CrossEntropy<br>- Scheduler: ReduceLROnPlateau", 540, 360, 200, 100, style_dl_dark))
    nodes.append(create_node("dl_eval", "<b><font style='font-size: 14px'>Validation Loss</font></b><br>Decreasing?", 780, 360, 120, 100, style_diamond))
    nodes.append(create_node("check_pat", "<b><font style='font-size: 14px'>Patience < 10?</font></b>", 950, 380, 120, 60, style_diamond))
    nodes.append(create_node("dl_stop", "<b><font style='font-size: 14px'>Early Stopping</font></b><br>(Stop Training)", 1120, 380, 120, 60, style_stop))
    nodes.append(create_node("dl_save", "<b><font style='font-size: 14px'>Save Best Model</font></b><br>(.pth)", 780, 500, 120, 60, style_stop))

    # --- Edges ---
    edges.append(create_edge("e1", "raw", "prep"))
    edges.append(create_edge("e2", "prep", "merge"))
    edges.append(create_edge("e3", "merge", "split"))
    
    # Split to Loader (Cross Row)
    edges.append(f"""<mxCell id="e4" value="Scaled Sequences" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;entryX=0.5;entryY=0;" edge="1" parent="1" source="split" target="dl_loader">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>""")

    edges.append(create_edge("e5", "dl_loader", "cnn_arch"))
    edges.append(create_edge("e6", "cnn_arch", "dl_train"))
    edges.append(create_edge("e7", "dl_train", "dl_eval"))
    
    # Eval Logic
    edges.append(f"""<mxCell id="e8" value="Yes (New Best)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="dl_eval" target="dl_save">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>""")
    
    edges.append(create_edge("e9", "dl_eval", "check_pat", "No"))
    
    # Patience Logic
    edges.append(create_edge("e10", "check_pat", "dl_stop", "No (Stop)"))
    
    # Loop back (Continue)
    edges.append(f"""<mxCell id="e11" value="Yes (Continue)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=1;entryX=0.5;entryY=1;" edge="1" parent="1" source="check_pat" target="dl_train">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="1010" y="480" />
              <mxPoint x="640" y="480" />
            </Array>
          </mxGeometry>
        </mxCell>""")

    # Save Loop back
    edges.append(f"""<mxCell id="e12" value="Next Epoch" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0;exitY=0.5;entryX=0.5;entryY=1;" edge="1" parent="1" source="dl_save" target="dl_train">
          <mxGeometry relative="1" as="geometry">
             <Array as="points">
              <mxPoint x="640" y="530" />
            </Array>
          </mxGeometry>
        </mxCell>""")

    # XML Construction
    xml_content = f"""<mxfile host="app.diagrams.net">
  <diagram name="Flowchart">
    <mxGraphModel dx="1400" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {"".join(nodes)}
        {"".join(edges)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

    with open("flowchart.drawio", "w") as f:
        f.write(xml_content)
    print("Generated flowchart.drawio")

if __name__ == "__main__":
    generate_dot()
    generate_mermaid()
    generate_drawio()

