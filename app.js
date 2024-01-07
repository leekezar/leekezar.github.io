// Function to determine if an element is in the viewport
function inViewport($element) {
    var elementTop = $element.offset().top;
    var elementBottom = elementTop + $element.outerHeight();
    var viewportTop = $(window).scrollTop();
    var viewportBottom = viewportTop + $(window).height();
    if (elementTop > viewportTop && elementBottom < viewportBottom) {
        return 1;
    }
    return 0;
}

$(document).ready(function(){
    var startDate = new Date('2019-08-01');
    var endDate = new Date('2024-12-31');
    var currentDate = new Date();
    var progress = ((currentDate - startDate) / (endDate - startDate)) * 100;
    $("#progressbar").progressbar({ value: progress });
    $(".progress-label").text("Loading PhD... (" + progress.toFixed(0) + "%)");

    // Script for expanding concepts and papers
    $(".expandable").click(function(){
        var clickedId = $(this).attr('id');
        var objToExpand = $(".expansion#" + clickedId)
        var parentToExpand = objToExpand.parent(".expansion");
        var parentId = parentToExpand.attr("id");

        if (clickedId === "collapse_concepts"){
            $(".expansion.concept").slideUp("slow");
            $(".expandable.concept").removeClass("expanded_highlight");

        } else if (clickedId === "expand_concepts") {
            $(".expansion.concept").slideDown("slow");
            $(".expandable.concept:not(#collapse,#expand)").addClass("expanded_highlight");
            $(".expandable.concept#collapse .expandable#expand").removeClass("expanded_highlight");

        } else if (clickedId === "collapse_papers"){
            $(".expansion.paper").slideUp("slow");
            $(".expandable.paper").removeClass("expanded_highlight");

        } else if (clickedId === "expand_papers") {
            $(".expansion.paper").slideDown("slow");
            $(".expandable.paper:not(#collapse,#expand)").addClass("expanded_highlight");
            $(".expandable.paper#collapse .expandable#expand").removeClass("expanded_highlight");

        } else if (!$(".expandable#" + clickedId).hasClass("expanded_highlight")) { // Item is not shown; show it.
            // Highlight the clicked button
            $(".expandable#" + clickedId).addClass("expanded_highlight");
            
            // Optionally show the parent if it isn't already shown
            if (!$(".expandable#" + parentId).hasClass("expanded_highlight") && parentToExpand.is("section")) {
                // Blink it a couple times to show the user
                // Instead of using fadeIn / fadeOut, add and remove the class "expanded_highlight" a few times with a brief delay between
                var blink_speed = 100;
                var blink_count = 2;
                var blink_interval = 80;
                var blink_index = 0;
                var blink_function = function() {
                    if (blink_index < blink_count) {
                        $(".expandable#" + parentId).addClass("expanded_highlight");
                        setTimeout(function() {
                            $(".expandable#" + parentId).removeClass("expanded_highlight");
                        }, blink_speed);
                        blink_index++;
                        setTimeout(blink_function, blink_speed + blink_interval);
                    } else {
                        $(".expandable#" + parentId).addClass("expanded_highlight");
                        parentToExpand.slideDown("slow", function() {
                            objToExpand.slideDown("slow");
                            // Scroll to the clicked button only if it's not in the viewport
                            var scrollAmount = objToExpand.offset().top + (objToExpand.height() * 1.2) - $(window).height();
                            if (inViewport(objToExpand) === 0) {
                                $("html, body").animate({
                                    scrollTop: scrollAmount
                                }, 500);
                            }
                        });
                    }
                };

                blink_function();
            } else {
                $(".expandable#" + clickedId).addClass("expanded_highlight");
                objToExpand.slideDown("slow", function() {
                    // Scroll to the clicked button
                    var scrollAmount = objToExpand.offset().top + (objToExpand.height() * 1.2) - $(window).height();
                    if (inViewport(objToExpand) === 0) {
                        $("html, body").animate({
                            scrollTop: scrollAmount
                        }, 500);
                    }
                });
            }

        } else { // Item is shown, hide it and its children, and remove expanded_highlight class
            $(".expandable#" + clickedId).removeClass("expanded_highlight");
            $(".expansion.super#" + clickedId).find(".expanded_highlight:not(.super,.paper)").each(function() {
                var id = $(this).attr('id');
                $(".expandable#" + id).removeClass("expanded_highlight");
                $(".expansion#" + id).slideUp("slow");
            });

            objToExpand.slideUp("slow");
        }
    });
});