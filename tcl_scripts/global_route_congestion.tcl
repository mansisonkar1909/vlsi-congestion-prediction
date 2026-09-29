# global_route_congestion.tcl
# Run this INSIDE the OpenROAD docker/app, e.g.:
#   openroad tcl_scripts/global_route_congestion.tcl
#
# It reloads the placed design, runs global routing, and dumps a
# congestion report (per-gcell overflow) to CSV. This becomes your
# "ground truth" heatmap (Step 4).

# ---- 1. Reload the placed design ----
read_db ./data/raw/placed.odb

# ---- 2. Set up routing layers (EDIT for your tech) ----
set_global_routing_layer_adjustment metal2-metal10 0.5
set_routing_layers -signal metal2-metal10

# ---- 3. Run global routing ----
global_route -congestion_report_file ./data/raw/congestion_report.csv \
              -congestion_report_iter_step 1 \
              -overflow_iterations 50

# The congestion_report.csv produced by global_route already contains,
# per GCell, columns roughly like:
#   layer, x1, y1, x2, y2, capacity, usage, overflow
# We rely on OpenROAD's built-in report rather than hand-parsing routes.

puts "Global routing + congestion report complete."
