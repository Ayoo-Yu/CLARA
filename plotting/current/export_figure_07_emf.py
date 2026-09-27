"""Preserve hatching as vector line segments for Word's EMF renderer."""
from pathlib import Path
from lxml import etree as ET
import re, math, subprocess, struct, json, argparse, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inkscape', help='Path to the Inkscape executable; defaults to PATH lookup.')
parser.add_argument('--directory', type=Path, help='Directory containing Figure_07.svg; defaults to outputs/current/figure07.')
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
HERE = args.directory or REPO / 'outputs/current/figure07'
ink = args.inkscape or shutil.which('inkscape') or shutil.which('inkscape.com')
if not ink:
    parser.error('Inkscape was not found. Install it or provide --inkscape.')
ns = {'s': 'http://www.w3.org/2000/svg'}
tag = '{' + ns['s'] + '}'
tree = ET.parse(str(HERE / 'Figure_07.svg'))
count = 0
for element in list(tree.xpath('//s:path[contains(@style,"fill: url(#h")]', namespaces=ns)):
    original_style = element.get('style')
    pattern_id = re.search(r'fill: url\(#([^\)]+)\)', original_style).group(1)
    pattern = tree.xpath('//s:pattern[@id=$pid]', namespaces=ns, pid=pattern_id)[0]
    fill = pattern.find(tag+'rect').get('fill')
    coords = list(map(float, re.findall(r'-?\d+(?:\.\d+)?', element.get('d'))))
    assert len(coords) == 8
    xmin, xmax = min(coords[0::2]), max(coords[0::2])
    ymin, ymax = min(coords[1::2]), max(coords[1::2])
    parent = element.getparent()
    position = parent.index(element)
    background = ET.Element(tag+'path', d=element.get('d'), style=f'fill: {fill}; stroke: none')
    parent.insert(position, background)
    position += 1
    # The saved Matplotlib /// pattern has x+y=8k and 0.45 pt strokes.
    for k in range(math.ceil((xmin+ymin)/8), math.floor((xmax+ymax)/8)+1):
        total = k*8
        points = []
        for x, y in [(xmin,total-xmin),(xmax,total-xmax),(total-ymin,ymin),(total-ymax,ymax)]:
            if xmin-1e-8 <= x <= xmax+1e-8 and ymin-1e-8 <= y <= ymax+1e-8:
                if not any(abs(x-px)<1e-8 and abs(y-py)<1e-8 for px,py in points):
                    points.append((x,y))
        if len(points)==2:
            (x1,y1),(x2,y2) = points
            hatch = ET.Element(tag+'path', d=f'M {x1} {y1} L {x2} {y2}',
                style='fill: none; stroke: #000000; stroke-width: 0.45; stroke-linecap: butt')
            parent.insert(position,hatch)
            position += 1
    element.set('style', re.sub(r'fill: url\(#[^\)]+\)', 'fill: none',original_style))
    count += 1
assert count == 2
compat = HERE/'Figure_07_emf_compat.svg'
outlined = HERE/'Figure_07_emf_outlined.svg'
tree.write(str(compat),encoding='utf-8',xml_declaration=True)
for args in [[compat,'--export-text-to-path','--export-plain-svg','--export-filename='+str(outlined)],
             [outlined,'--export-type=emf','--export-filename='+str(HERE/'Figure_07.emf')]]:
    subprocess.run([ink,*map(str,args)],capture_output=True,check=True)
data = (HERE/'Figure_07.emf').read_bytes()
records = []
pos = 0
while pos < len(data):
    kind,size = struct.unpack_from('<II',data,pos)
    assert size >= 8 and pos+size <= len(data)
    records.append(kind)
    pos += size
assert records[0] == 1 and records[-1] == 14
bitmap_records = {76,77,78,79,80,81,114,116}
assert not bitmap_records.intersection(records)
audit = {'status':'PASS','record_count':len(records),'bitmap_records':[],
         'hatched_regions_expanded_to_vector_lines':count,
         'emf_text':'Outlined for font preservation; editable text retained in Figure_07.svg and .pdf.'}
(HERE/'emf_qa.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit))
