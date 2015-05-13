var w = window.innerWidth/1.2;
var h = window.innerHeight/1.5;
var draw, map, legend;
var projection = d3.geo.mercator();
    projection.scale([w/3.5]).translate([w/4,h/1.9])

var path = d3.geo.path().projection(projection);

var svg = d3.select("article")
	.append("svg")
	.attr({
		width: w, 
		height: h
	})

colorRange= d3.scale.category20b().range().reverse();
colorRange.splice(6,1);
colorRange.splice(0,1);
colorRange.splice(11,1);
colorRange.splice(14,1);
colorRange.splice(1,1);
colorRange.splice(7,1);

var color = d3.scale.quantize()
    .range(colorRange)
    .domain([-15.10, 38.00]); 

legend = d3.select('#legend').append('ul');
	    		
	
d3.json("../Data/africa.json", function(json){
	d3.csv("../Data/gdpGrowth.csv",function(csv){
	
		draw = function(year){
			
			for(var j = 0; j < csv.length; j++){
				for(var i = 0; i < json.features.length; i++){
						if(json.features[i].properties.name.toLowerCase() == csv[j].country){
							json.features[i].properties.gdpGrowth = csv[j][year];
							break;
						}
					}
				}

		
	
			 map = svg.selectAll("path")
				.data(json.features)
				.enter()
				.append("path")
				.attr("d", path)
	            .style({
	            	"fill":function(d) {
	               	  var value = d.properties.gdpGrowth;
               		  if (value) { return color(value); }
                      else { return "#ffffff"; }},
	                 "opacity":.8
	            })
	            .text(function(d){
	            	return d.properties.name;
	            })
				.on("mouseover", function(d){	
					var coordinates = d3.mouse(this);
					d3.select(this).style("opacity",1)			
					d3.select("#tooltip")
					.style({
							"left": coordinates[0]  + "px",
							"top": coordinates[1] + "px"
						}).classed("hidden",false)
						.select("#gdpGrowth").append("text")
						.text(function(){
							if(d.properties.gdpGrowth){
								return d.properties.admin + ", GDP growth from " + String(year-1) + ": " + d.properties.gdpGrowth + "%";
							}
						})
				})
				.on("mouseout",function(d){
					d3.select(this).style("opacity",.8)
					d3.select("#tooltip").classed("hidden",true).select("text").remove();
				})
			
	    	
	    	var keys = legend.selectAll('li')
	    		.data(color.range());
	    	
	    	keys.enter().append('li').classed("legend",true)
	    		.style({
	    			"border-top-color":String,
	    			"opacity": 1,
	    			"width":'10px'

	    		})
	    		.text(function(d) {
	        		var r = color.invertExtent(d);
	        		return Math.ceil(r[0]) +" - " + Math.floor(r[1]) ; 
	    		});
			
		}

		draw(2012);
		
		$(".btn").click(function(e){
			 $("#yearSelect").find(".highlight").removeClass("highlight");
			e.preventDefault();
			map.remove();
			draw($(this).attr('data-value'));
		 	$(this).addClass("highlight");
		
		});
   			
	})
		
})