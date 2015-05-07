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
	
var color = d3.scale.quantize()
    .range(['rgb(255,255,217)','rgb(237,248,177)','rgb(199,233,180)','rgb(127,205,187)','rgb(65,182,196)','rgb(29,145,192)','rgb(34,94,168)','rgb(37,52,148)'])
    .domain([0.0, 55.00]); // starting at 0, because there are some regions I have no data on..

legend = d3.select('#legend').append('ul');
	    		
	
d3.json("africa.json", function(json){
	d3.csv("../Data/gdpGrowth.csv",function(csv){
	
		draw = function(year){
			
			for(var j = 0; j < csv.length; j++){
				console.log(j)
				for(var i = 0; i < json['data'].geo.features.length; i++){
						if(json['data'].geo.features[i].properties.name.toLowerCase() == csv[j].country){
							console.log("yes")
							json['data'].geo.features[i].properties.perCent = csv[j][year];
							break;
						}				
					}
				}
		
	
			 map = svg.selectAll("path")
				.data(json['data'].geo.features)
				.enter()
				.append("path")
				.attr("d", path)
	            .style({
	            	"fill":function(d) {
	               	  var value = d.properties.perCent;
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
						.select("#perCent").append("text")
						.text(function(){
							if(d.properties.perCent){
								return d.properties.name + ", " + d.properties.perCent + "%";
							}
							else{
								return d.properties.name + ", UN Data not available";
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
	    			"opacity": .8
	    		})
	    		.text(function(d) {
	        		var r = color.invertExtent(d);
	        		return Math.ceil(r[0]) +" - " + Math.floor(r[1]) ; // give the approximate value
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