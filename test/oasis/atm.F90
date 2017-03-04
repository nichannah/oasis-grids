
program atm

use coupler, only : coupler_init, coupler_init_done, coupler_add_field, &
                    coupler_destroy_field, coupler_put, coupler_close, &
                    coupler_destroy_field, couple_field_type, &
                    COUPLER_OUT, coupler_dump_field
implicit none

    type(couple_field_type), dimension(:), allocatable :: fields
    integer :: i, j, t, timestep
    logical :: debug

    debug = .false.
    timestep = 1

    ! Initialise the coupler.
    call coupler_init('atmxxx', 192, 94, 1, 1)

    ! Create/add the coupling field.
    allocate(fields(1))
    call coupler_add_field(fields(1), 'src_field', COUPLER_OUT)
    call coupler_init_done()

    ! Initialise the field.
    do j=1, size(fields(1)%field, 2)
      do i=1, size(fields(1)%field, 1)
        fields(1)%field(i, j) = j
      enddo
    enddo
    if (debug) then
      call coupler_dump_field(fields(1), 'src_field.nc')
    endif

    do t=1,100
      print*, 'Timestep: ', t
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
      call coupler_put(timestep*t, fields)
    enddo

    call coupler_destroy_field(fields(1))
    call coupler_close()

    deallocate(fields)

end program atm
