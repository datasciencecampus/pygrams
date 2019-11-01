var dataURL = "https://raw.githubusercontent.com/datasciencecampus/patent_app_detect/master/outputs/visuals/empty.json";

var refresh = function(data){

var json_obj = JSON.parse(data);
var svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var graph = json_obj[0];

    d3.selectAll("g > *").remove();
//TODO make svg responsive
d3.select("div#chartId")
    .append("div")
    .classed("svg-container", true) //container class to make  responsive svg
    .append("svg")
    //responsive SVG needs these 2 attributes and no width and height attr
    .attr("preserveAspectRatio", "xMinYMin meet")
    .attr("viewBox", "0 0 9000 6000")
    .attr("overflow", "scroll")
    //class to make it responsive
    .classed("svg-content-responsive", true);

var color = d3.scaleOrdinal(d3.schemeCategory20c);

var padding = 1, // separation between circles
    radius=6;
	
var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function(d) {
        return d.text;
    }).distance(300))
    .force("charge", d3.forceManyBody().strength(-100))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(function(d) {
        return 12 * radius + padding;
    }).iterations(40));

d3.json(dataURL, function(error, graph) {
    if (error) throw error;

	graph = json_obj[0];
    simulation.nodes(graph.nodes);
    simulation.force("link").links(graph.links);

    var link = svg.append("g")
        .attr("class", "link")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line").attr("stroke-width", function(d) {
        return (8*d.size);
        });

    var node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")

    //Setting node radius by group value. If 'group' key doesn't exist, set radius to 9
    .attr("r", function(d) {
            if (d.hasOwnProperty('freq')) {
                return d.freq * 40;
            } else {
                return 9;
            }
        })
        //Colors by 'group' value
        .style("fill", function(d) {
            return color(d.freq * 20);
        })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
			); //Added code 

    node.append("svg:title")
        .attr("dx", 18)
        .attr("dy", ".35em")
        .text(function(d) {
            return d.id
        });

    

    var labels = svg.append("g")
        .attr("class", "label")
        .selectAll("text")
        .data(graph.nodes)
        .enter().append("text")
        .attr("dx", 6)
        .attr("dy", ".35em")
        .style("font-size", 18)
        .text(function(d) {
            return d.text
        });

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);
  
  function ticked() {

    link.attr("x1", function(d) {
            return d.source.x;
        })
        .attr("y1", function(d) {
            return d.source.y;
        })
        .attr("x2", function(d) {
            return d.target.x;
        })
        .attr("y2", function(d) {
            return d.target.y;
        });

    node.attr("cx", function(d) { return d.x = Math.max(radius, Math.min(width - radius, d.x)); })
        .attr("cy", function(d) { return d.y = Math.max(radius, Math.min(height - radius, d.y)); });
    labels
        .attr("x", function(d) {
            return d.x;
        })
        .attr("y", function(d) {
            return d.y;
        }); 
}
 
})
;

function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
}

function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}
};

refresh(data);
