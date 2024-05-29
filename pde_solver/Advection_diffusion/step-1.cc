#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
 
#include <deal.II/numerics/data_out.h>
 
#include <fstream>
#include <iostream>
 
#include <deal.II/numerics/vector_tools.h>
 
#include <deal.II/numerics/matrix_tools.h>
 
#include <deal.II/base/utilities.h>

#include <deal.II/base/hdf5.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>


#include <sstream>
#include <string>


#include <deal.II/grid/grid_out.h>
#include <math.h>

 
namespace Step23
{
  using namespace dealii;
 
 
 
  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation();
    void run();
 
  private:
    void setup_system();
    void solve();
    void output_results() const;
 
    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;
 
    AffineConstraints<double> constraints;
 
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
 
    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;
 
    double       time_step;
    double       time;
    unsigned int timestep_number;
    const double theta;
  };
 
 
 
 
  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      if ((p[0]> 3500000. && p[0] < 3600000.) && (p[1] > 5400000. && p[1] < 5500000.))
	      {
		      return 1;
	      }
      else
	return 0;
      //return 0;
    }
  };
   
template <int dim>
  class Diffusion : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return -1.;
    }
  };
 

  template <int dim>
  class RightHandSide : public Function<dim>//source terms
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      /*Gaussian kernel
      double sigma_squared = 10000000000;
      double x_diff = p[0] - 3600000.; 
      double y_diff = p[1] - 5400000.;
      double exponent = -(x_diff * x_diff + y_diff * y_diff) / (2.0 * sigma_squared);
      return 0.3*std::exp(exponent);*/
      float time_component = ((this->get_time()*1e-8-2.5))/97.5;//from 0 to 1
	float modulo_time_component = fmod((time_component/3),6);
	float center_y;// = 3.5e6 *(1-time_component) + 3.6e6 *(time_component) ; //= 3600000.
	float center_x;// = 5.4e6 *(1-time_component) + 5.5e6 *(time_component); //5400000.
	if (modulo_time_component <= 1)//TODO
      {
		center_y = 3.45e6 ; //= 3600000.
		center_x = 5.4e6; //5400000.
	}

	else if ((modulo_time_component > 1) && (modulo_time_component <= 2))//TODO
      {
		center_y = 3.45e6 *(2 - modulo_time_component) + 3.6e6 *(modulo_time_component-1) ; //= 3600000.
		center_x = 5.4e6 *(2 - modulo_time_component) + 5.5e6 *(modulo_time_component-1); //5400000.
	}
	else if ((modulo_time_component > 2) && (modulo_time_component <= 3))//TODO
      {
		center_y = 3.6e6 ; //= 3600000.
		center_x = 5.6e6; //5400000.
	}
      else if((modulo_time_component > 3) && (modulo_time_component <=4)){
		center_y = 3.6e6 *(4 - modulo_time_component) + 3.45e6 *(modulo_time_component-3) ; //= 3600000.
		center_x = 5.6e6 *(4 - modulo_time_component) + 5.8e6 *(modulo_time_component-3); //5400000.
      }
	else if ((modulo_time_component > 4) && (modulo_time_component <= 5))//TODO
      {
		center_y = 3.45e6 ; //= 3600000.
		center_x = 5.8e6; //5400000.
	}
     else if(( modulo_time_component > 5) && (modulo_time_component <=6)){
		center_y = 3.45e6 *(6-modulo_time_component) + 3.45e6 *(modulo_time_component-5) ; //= 3600000.
		center_x = 5.8e6 *(6-modulo_time_component) + 5.4e6 *(modulo_time_component-5); //5400000.
      }


      if ((p[0]> center_y && p[0] < center_y * 1.01) && (p[1] > center_x && p[1] < center_x * 1.01))
      //if ((p[0]> 3600000. && p[0] < 3700000.) && (p[1] > 5400000. && p[1] < 5500000.))
	      {
		      if(fmod(time_component ,3) <=1)
		      {
			      return -(std::sin(modulo_time_component *5*numbers::PI)*16 + 16.0)*1e-10;
		      }
		      else if((fmod(time_component ,3) > 1) && (fmod(time_component ,3) <=2)) 
		      {
			      return -(std::sin(modulo_time_component *2*numbers::PI)*10 + 12)*1e-10;
		      }
		      else if((modulo_time_component > 2) && (modulo_time_component <=3))
		      {
			      return -(std::sin(modulo_time_component *3*numbers::PI)*13 + 13)*1e-10; //this varies between 6.5 and 10.5 
		      }
		      return -7e-10;
	      }
      else
	return 0;
      //return 0.;
    }
  };


template <int dim>
    class AdvectionField : public TensorFunction<1, dim>
    {
    public:
      virtual Tensor<1, dim> value(const Point<dim> &p) const override;
    };
  
    template <int dim>
    Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
    {
      Tensor<1, dim> value;
      //std::cout<< p[1]*1e-6<< " gives: "<<((p[1]*1e-6*5)-27.5)<<std::endl;
      float time_factor = -(this->get_time()*1e-8*(-0.02))-1; //-1 to 1
      float time_component = ((this->get_time()*1e-8-2.5))/97.5;
      if ((time_component/18) <= 1)
      {
		value[0] = 0.00005*(1);//*time_factor;//2; this is y-axis // ((std::cos(2* time_component * numbers::PI)) -1)
		value[1] = -0.00005;
      }
      else if (((time_component/18) >1) && ((time_component/18) <=2))
      {
		value[0] = -0.00002*((p[1]*1e-6*5)-27.5);//*time_factor;//2; this is y-axis
		value[1] = 0.00003* (2);
      }
      else if (((time_component/18) >2) && ((time_component/18) <=3))
      {
		value[0] = 0.00005*((p[1]*1e-6*5)-27.5);//*time_factor;//2; this is y-axis
		value[1] = 0.00005;
      }
      return value;
    }
 
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0.;//50000 * std::sin(this->get_time() * 0.00004 * numbers::PI);
    }
  };
 
 
 
 
 
  template <int dim>
  WaveEquation<dim>::WaveEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1000000000./8) //1e9
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
  {}
 
 
 
  template <int dim>
  void WaveEquation<dim>::setup_system()
  {
    //GridGenerator::hyper_cube(triangulation, -1, 1);
    //triangulation.refine_global(7);
    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("Germanybloate_.msh");
    gridin.read_msh(f);

    std::ofstream out("germany_grid.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
 
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
 
    dof_handler.distribute_dofs(fe);
 
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;
 
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
 
    system_matrix.reinit(sparsity_pattern);
 
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
 
    constraints.close();
  }
 
 
 
 
  template <int dim>
  void WaveEquation<dim>::solve()
  {

    SparseDirectUMFPACK A_direct;
    A_direct.solve(system_matrix, system_rhs);
    solution = system_rhs;
 
  }
 
 
 
  template <int dim>
  void WaveEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;
 
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(old_solution, "U");
 
    data_out.build_patches();
 
    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtk(output);
   
    Vector<double> point_value(1);
    const std::string filename_h5 = "solution-" + std::to_string(1) + "_"  + Utilities::int_to_string(timestep_number, 4) +   ".csv";
    std::ofstream csvfile;
    csvfile.open(filename_h5);

    csvfile<<"Value \n";

    std::vector<std::vector<double>> nuts_coords;
    
    // Read the CSV file and extract the X and Y coordinates
    std::ifstream file("germany_centers.csv");
    
    if (file.is_open())
    {
	    std::string line;
	    std::getline(file, line); // Read the header line and ignore it
	    
	    
	    while (std::getline(file, line))
	    {
		    std::istringstream iss(line);
		    std::string token;
		    std::vector<double> coordinates;
		    
		    for (int i = 0; i < 3; ++i) // Read all four columns
		    {
			    std::getline(iss, token, ',');
			    if (i >= 1) // Extract X and Y coordinates from 3rd and 4th columns
			    {
				    coordinates.push_back(std::stod(token));
			    }
		    }
		    nuts_coords.push_back(coordinates);
	    }
	    file.close();
    }
    else
    {
	    std::cerr << "Unable to open file germany_centers.csv" << std::endl;
    	    // Handle error if needed
    }




    for (int i = 0; i < 400; i ++)
    {
	    try{
	    VectorTools::point_value(dof_handler, old_solution, Point<2>(nuts_coords[i][0], nuts_coords[i][1]), point_value);
	    //csvfile<<point_value[0]<< ", "<< point_value[1]<<""<<"\n";
	    csvfile<<point_value[0] <<""<<"\n";
	    }
	    catch (...)
	    {
		    std::cout<< "Not working on " << i << std::endl;
		    std::cout<< "Points there are:  " << nuts_coords[i][0] << "and: " <<nuts_coords[i][1]<< std::endl;
	    }
    }
    csvfile.close();
  }





 
 
 
  template <int dim>
  void WaveEquation<dim>::run()
  {
    setup_system();
 
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValues<dim>(),
                         old_solution);
     QGauss<2> quadrature_formula(fe.degree + 1);
     FEValues<2> fe_values(fe,
		     quadrature_formula,
		     update_values | update_quadrature_points | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    RightHandSide<dim> rhs_function;
    RightHandSide<dim> old_rhs_function;
    Diffusion<dim> diffusion_alpha;
    AdvectionField<dim> advection_field;
    AdvectionField<dim> old_advection_field;
 
    for (; time <= 54*1e10; time += time_step, ++timestep_number) //54 = 3*6*3
      {
	      rhs_function.set_time(time);
	      old_rhs_function.set_time(time + time_step);//actually not old, but future
	      //diffusion_alpha.set_time(time); //implicit and explicit at different times?
	      advection_field.set_time(time);//
	      old_advection_field.set_time(time+time_step);//
	      system_matrix = 0;
	      system_rhs = 0;
	      
	      for (const auto &cell : dof_handler.active_cell_iterators())
	      {
		      fe_values.reinit(cell);
		      std::vector<Tensor<1, dim>> old_solution_gradients(fe_values.n_quadrature_points);
		      fe_values.get_function_gradients(old_solution, old_solution_gradients);

		      std::vector<double> old_solution_values(fe_values.n_quadrature_points);
		      fe_values.get_function_values(old_solution, old_solution_values);
		
			cell_matrix = 0;
			cell_rhs = 0;
			//system_rhs.add(theta * time_step, forcing_terms);
			for (const unsigned int q_index : fe_values.quadrature_point_indices())
			{
				const auto &x_q = fe_values.quadrature_point(q_index);
				for (const unsigned int i : fe_values.dof_indices())
					for (const unsigned int j : fe_values.dof_indices())
					{
					cell_matrix(i,j) += ((fe_values.shape_value(i, q_index)
						* fe_values.shape_value(j, q_index))
						- 
						(time_step
						* (1-theta)
						* fe_values.shape_value(i, q_index)
						* (advection_field.value(x_q)
							* fe_values.shape_grad(j,q_index)))
						-(
						time_step
						* (1-theta)
						* diffusion_alpha.value(x_q)//TODO  check?
						* fe_values.shape_grad(i, q_index)
						* fe_values.shape_grad(j, q_index))
						)*fe_values.JxW(q_index);

					}
				for (const unsigned int i : fe_values.dof_indices())
				{
					
					cell_rhs(i) += ((fe_values.shape_value(i, q_index)
						* old_solution_values[q_index])
						+
						(time_step
						* theta
						* fe_values.shape_value(i, q_index)
						* (old_advection_field.value(x_q) 
								* old_solution_gradients[q_index]))
						+ 
						(time_step 
						* diffusion_alpha.value(x_q)//TODO  check?
						* theta
						* fe_values.shape_grad(i,q_index)
						*old_solution_gradients[q_index])
						-
						(time_step
						* theta
						* fe_values.shape_value(i, q_index)
						* rhs_function.value(x_q))
						-
						(time_step
						* (1-theta)
						* fe_values.shape_value(i, q_index)
						* old_rhs_function.value(x_q)))
						* fe_values.JxW(q_index);
									     
				}

			}
			cell->get_dof_indices(local_dof_indices);
			for (const unsigned int i : fe_values.dof_indices())
			{
				for (const unsigned int j : fe_values.dof_indices())
					system_matrix.add(local_dof_indices[i],
							local_dof_indices[j],
							cell_matrix(i, j));
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}

	}
	//step3 does something with interpolate_boundary_values
	std::map<types::global_dof_index, double> boundary_values;
        BoundaryValues<dim> boundary_values_function;
        boundary_values_function.set_time(time);

	VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_values_function, boundary_values);
	MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,//TODO maybe old_solution?
                                             system_rhs);
                
        solve();
 	std::cout<<"Solved at time t="<<time<<std::endl;
        output_results();
        old_solution = solution;
      }
  }
} // namespace Step23
 
 
 
int main(int argc, char* argv[])
{
  try
    {
      using namespace Step23;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
 
      WaveEquation<2> wave_equation_solver;
      wave_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}
