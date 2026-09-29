# extract_placement.tcl
# Run this INSIDE the OpenROAD docker/app, e.g.:
#   openroad tcl_scripts/extract_placement.tcl
#
# It loads a design, runs global + detailed placement,
# then dumps every cell's position and every pin's position to CSV.
# These CSVs are later turned into the "input image" (Step 3).

# ---- 1. Load technology + design (EDIT THESE PATHS for your design) ----
read_lef   ./data/raw/tech.lef
read_lef   ./data/raw/cells.lef
read_def   ./data/raw/floorplan.def
read_liberty ./data/raw/cells.lib

# ---- 2. Run placement ----
global_placement
detailed_placement

# ---- 3. Dump cell (instance) positions to CSV ----
set fh [open "./data/raw/cell_positions.csv" w]
puts $fh "name,x,y,width,height"

set block [ord::get_db_block]
foreach inst [$block getInsts] {
    set bbox [$inst getBBox]
    set name [$inst getName]
    set x [$bbox xMin]
    set y [$bbox yMin]
    set w [expr {[$bbox xMax] - [$bbox xMin]}]
    set h [expr {[$bbox yMax] - [$bbox yMin]}]
    puts $fh "$name,$x,$y,$w,$h"
}
close $fh

# ---- 4. Dump pin positions to CSV ----
# NOTE: OpenROAD's Tcl/odb API changes slightly between versions.
# Check `odb::dbITerm` docs for your version to get exact accessor names
# (e.g. getBBox, getAvgXY). The pattern below is the general idea:
set fh2 [open "./data/raw/pin_positions.csv" w]
puts $fh2 "inst_name,pin_name,x,y"

foreach inst [$block getInsts] {
    set iname [$inst getName]
    foreach iterm [$inst getITerms] {
        set mterm [$iterm getMTerm]
        set pname [$mterm getName]
        set bbox [$iterm getBBox]
        set px [expr {([$bbox xMin] + [$bbox xMax]) / 2}]
        set py [expr {([$bbox yMin] + [$bbox yMax]) / 2}]
        puts $fh2 "$iname,$pname,$px,$py"
    }
}
close $fh2

# ---- 5. Save the placed design so the routing script can reload it ----
write_def ./data/raw/placed.def
write_db  ./data/raw/placed.odb

puts "Placement extraction complete."
